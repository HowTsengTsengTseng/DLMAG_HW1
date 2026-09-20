"""Extract frozen MERT features, then fit one task-specific MLP."""
import argparse
from pathlib import Path

from dotenv import load_dotenv
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import check_disjoint, read_manifest
from features import extract_features
from models import MERTEncoder, MLPClassifier
from utils import LABELS, MODEL_ID, get_device, metrics, save_confusion, save_json, seed_everything


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=LABELS)
    parser.add_argument("--data-root", default="data/raw", help="Root directory containing datasets (default: data/raw)")
    parser.add_argument("--manifest", help="Combined manifest CSV (e.g. data/raw/dataset_A/manifest.csv); auto-filters splits")
    parser.add_argument("--train-manifest", help="Train manifest CSV (defaults to --manifest)")
    parser.add_argument("--val-manifest", help="Validation manifest CSV (defaults to --manifest)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="Hugging Face revision; resolved commit is saved")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.manifest:
        if not args.train_manifest:
            args.train_manifest = args.manifest
        if not args.val_manifest:
            args.val_manifest = args.manifest
    elif not (args.train_manifest and args.val_manifest):
        default_manifest = Path(args.data_root) / f"dataset_{args.task}" / "manifest.csv"
        if default_manifest.exists():
            args.train_manifest = str(default_manifest)
            args.val_manifest = str(default_manifest)
        else:
            parser.error("Must provide --manifest, both --train-manifest and --val-manifest, or have data/raw/dataset_{task}/manifest.csv")
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1:
        parser.error("epochs, patience, batch-size and hidden-dim must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid optimizer or dropout parameters")
    seed_everything(args.seed)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Prevent accidental reuse/overwrite of an existing run.
    if (output / "best.pt").exists():
        raise FileExistsError(f"Choose a new output directory: {output / 'best.pt'} exists")
    train_rows = read_manifest(args.train_manifest, args.data_root, args.task, "train")
    val_rows = read_manifest(args.val_manifest, args.data_root, args.task, "validation")
    check_disjoint(train_rows, val_rows)
    labels = LABELS[args.task]
    if {r["label"] for r in train_rows} != set(labels):
        raise ValueError("Training split must contain all six task labels")
    encoder = MERTEncoder(args.model_id, args.revision).to(device)
    revision, input_dim = encoder.revision, encoder.hidden_size
    features = [extract_features(rows, encoder, args.cache_dir, args.model_id,
                                 args.seconds, args.extract_batch_size) for rows in (train_rows, val_rows)]
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    x_train, x_val = features
    y_train, y_val = [torch.tensor([labels.index(r["label"]) for r in rows]) for rows in (train_rows, val_rows)]
    model_args = dict(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=len(labels), dropout=args.dropout)
    model = MLPClassifier(**model_args)
    model.fit_standardizer(x_train)  # Training samples only; buffers travel in the checkpoint.
    model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, stale, history = float("inf"), 0, []
    save_json(output / "config.json", vars(args) | {"resolved_revision": revision, "labels": labels})
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            logits = model(x_val.to(device)).cpu()
            val_loss = F.cross_entropy(logits, y_val).item()
        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError("Non-finite validation loss")
        scores = metrics(logits, y_val, labels)
        history.append({"epoch": epoch, "train_loss": total_loss / len(y_train), "val_loss": val_loss,
                        "top1": scores["top1"], "top3": scores["top3"]})
        print(history[-1], flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            checkpoint = {
                "format_version": 1, "task": args.task, "labels": labels,
                "model_id": args.model_id, "revision": revision, "seconds": args.seconds,
                "model_args": model_args, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "seed": args.seed,
                "train_ids": [r["sample_id"] for r in train_rows],
                "validation_ids": [r["sample_id"] for r in val_rows],
            }
            torch.save(checkpoint, output / "best.pt")
            save_json(output / "validation_metrics.json", scores | {"epoch": epoch, "loss": val_loss})
            save_confusion(scores, output / "validation_confusion.png")
        else:
            stale += 1
        save_json(output / "history.json", history)
        if stale >= args.patience:
            break
    print(f"Best checkpoint: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
