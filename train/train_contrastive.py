"""Fit one task-specific encoder and predict PURELY based on Era Contrastive (EC) Loss.

Training:
  - Optimizes the encoder solely using Supervised Contrastive Loss (Audio-SUC EC Loss, Eq. 3 in paper).
  - No Cross-Entropy loss is computed or backpropagated during training.

Prediction:
  - Class prototypes (mean embeddings on the unit hypersphere) are computed for each class/era.
  - Test/Validation predictions are made purely by ranking cosine similarities between query embeddings
    and class prototypes: score(x, c) = (z_x . w_c) / tau.
  - The final linear layer weights are fixed to these prototypes so that checkpoints remain 100% compatible
    with test.py inference.
"""
import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path when running from subfolder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import check_disjoint, read_manifest
from features import extract_features
from models import MERTEncoder, MLPClassifier
from utils import LABELS, MODEL_ID, get_device, metrics, save_confusion, save_json, seed_everything
from loss import EraContrastiveLoss


@torch.no_grad()
def compute_class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute unit-normalized mean centroid (prototype) for each class in contrastive space."""
    prototypes = torch.zeros((num_classes, embeddings.shape[1]), device=embeddings.device)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            class_mean = embeddings[mask].mean(dim=0)
            prototypes[c] = F.normalize(class_mean, p=2, dim=-1)
    return prototypes


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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    # EC contrastive loss temperature parameter
    parser.add_argument("--temperature", "--tau", type=float, default=0.1,
                        help="Temperature tau for EC contrastive loss (default: 0.1)")
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
    if args.temperature <= 0:
        parser.error("temperature must be positive")

    seed_everything(args.seed)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Prevent accidental overwrite of an existing run
    if (output / "best.pt").exists():
        raise FileExistsError(f"Choose a new output directory: {output / 'best.pt'} exists")

    train_rows = read_manifest(args.train_manifest, args.data_root, args.task, "train")
    val_rows = read_manifest(args.val_manifest, args.data_root, args.task, "validation")
    check_disjoint(train_rows, val_rows)
    labels = LABELS[args.task]
    num_classes = len(labels)
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

    model_args = dict(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, dropout=args.dropout)
    model = MLPClassifier(**model_args)
    model.fit_standardizer(x_train)
    model.to(device)

    # Pure EC loss criterion
    criterion_ec = EraContrastiveLoss(temperature=args.temperature)

    # Only train the encoder projection head (network[:3]). The final classifier weights (network[3])
    # are set directly by the contrastive prototypes.
    encoder_params = list(model.network[0].parameters())
    optimizer = torch.optim.AdamW(encoder_params, lr=args.lr, weight_decay=args.weight_decay)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))

    best_loss, stale, history = float("inf"), 0, []
    save_json(output / "config.json", vars(args) | {
        "resolved_revision": revision,
        "labels": labels,
        "objective": "pure_era_contrastive_loss",
        "prediction_rule": "nearest_prototype_cosine_similarity"
    })

    print(f"Training purely with Era Contrastive (EC) Loss (temperature={args.temperature})...")
    for epoch in range(1, args.epochs + 1):
        # --- TRAINING PHASE (Pure EC Loss) ---
        model.train()
        total_ec_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Extract normalized embedding z on unit hypersphere
            z = model.encode(x)
            loss = criterion_ec(z, y)

            loss.backward()
            optimizer.step()
            total_ec_loss += loss.item() * len(y)

        # --- PREDICTION & EVALUATION PHASE (Based on EC Prototypes) ---
        model.eval()
        with torch.no_grad():
            # 1. Compute class prototypes from all training embeddings
            z_train_all = model.encode(x_train.to(device))
            prototypes = compute_class_prototypes(z_train_all, y_train.to(device), num_classes)

            # 2. Assign prototypes to model.network[3] so model(x) computes prototype similarities
            model.set_prototypes(prototypes, temperature=args.temperature)

            # 3. Predict on validation set: logits = (z_val @ prototypes.T) / tau
            z_val = model.encode(x_val.to(device))
            logits_val = (torch.matmul(z_val, prototypes.T) / args.temperature).cpu()

            # Validation loss = negative log of prototype contrastive likelihood
            val_loss = F.cross_entropy(logits_val, y_val).item()

        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError("Non-finite validation loss")

        scores = metrics(logits_val, y_val, labels)
        train_ec_avg = total_ec_loss / len(y_train)
        record = {
            "epoch": epoch,
            "train_ec_loss": train_ec_avg,
            "val_loss": val_loss,
            "top1": scores["top1"],
            "top3": scores["top3"],
        }
        history.append(record)
        print(f"Epoch {epoch:03d} | Train EC Loss: {train_ec_avg:.4f} | Val Contrastive Loss: {val_loss:.4f} | Top-1: {scores['top1']*100:.2f}% | Top-3: {scores['top3']*100:.2f}%", flush=True)

        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            checkpoint = {
                "format_version": 1, "task": args.task, "labels": labels,
                "model_id": args.model_id, "revision": revision, "seconds": args.seconds,
                "model_args": model_args, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "seed": args.seed,
                "train_ids": [r["sample_id"] for r in train_rows],
                "validation_ids": [r["sample_id"] for r in val_rows],
                "training_loss": "pure_era_contrastive_loss",
                "temperature": args.temperature,
            }
            torch.save(checkpoint, output / "best.pt")
            save_json(output / "validation_metrics.json", scores | {"epoch": epoch, "loss": val_loss})
            save_confusion(scores, output / "validation_confusion.png")
        else:
            stale += 1

        save_json(output / "history.json", history)
        if stale >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"Best checkpoint saved to: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
