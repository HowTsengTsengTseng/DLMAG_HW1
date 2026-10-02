import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import AudioDataset, check_disjoint, read_manifest, LabeledAudioDataset
from loss import EraContrastiveLoss
from models import AudioSUCCNNv2, LinearClassifier
from utils import LABELS, get_device, metrics, save_confusion, save_json, seed_everything, collate_audio
from visualize import save_embedding_visualizations



def stage2(model, loader, val_loader, output, device, num_classes, args, bb_epoch, train_rows, val_rows, model_args):
    labels = LABELS[args.task]
    model.eval()  # freeze backbone
    best_val_loss, stale, history = float('inf'), 0, []
    clf = LinearClassifier(512, num_classes).to(device)
    clf_opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for clf_epoch in range(1, args.clf_epochs + 1):
        clf.train()
        train_loss_sum = 0
        for x, y in loader:
            h = model.encode(x.to(device)).to(device)
            y = y.to(device)
            clf_opt.zero_grad(set_to_none=True)
            logits = clf(h); loss = F.cross_entropy(logits, y); loss.backward(); clf_opt.step(); train_loss_sum += loss.item()

        with torch.no_grad():
            x_val, y_val = next(iter(val_loader))
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = clf(model.encode(x_val));
            val_loss = F.cross_entropy(logits, y_val).item() / len(x_val)

        scores = metrics(logits, y_val, labels)
        if val_loss < best_val_loss:
            best_val_loss, stale = val_loss, 0
            checkpoint = {
                "format_version": 1,
                "model_class": "AudioSUCCNNv2",
                "task": args.task,
                "labels": labels,
                "revision": "cnn_v2",
                "seconds": args.seconds,
                "model_args": model_args,
                "backbone_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "clf_state_dict": {key: value.detach().cpu() for key, value in clf.state_dict().items()},
                "seed": args.seed,
                "train_ids": [row["sample_id"] for row in train_rows],
                "validation_ids": [row["sample_id"] for row in val_rows],
                "temperature": args.temperature,
            }
            torch.save(checkpoint, output / "best.pt")
            save_json(output / "validation_metrics.json", scores | {"epoch": clf_epoch, "loss": val_loss})
            save_confusion(scores, output / "validation_confusion.png")
        else:
            stale += 1

        record = {
            "bb_epoch": bb_epoch,
            "clf_epoch": clf_epoch,
            "train_loss": train_loss_sum / len(loader),
            "val_loss": val_loss,
            "top1": scores["top1"],
            "top3": scores["top3"],
        }
        history.append(record)
        print(f"Clf Epoch {clf_epoch:03d} | Train {record['train_loss']:.4f} "
              f"| Val {val_loss:.4f} | Top-1 {scores['top1']*100:.2f}% "
              f"| Top-3 {scores['top3']*100:.2f}%", flush=True)

        save_json(output / "history.json", history)
        if stale >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=LABELS)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest", help="Combined manifest; train/validation rows are filtered by split")
    parser.add_argument("--train-manifest", help="Train manifest (defaults to --manifest)")
    parser.add_argument("--val-manifest", help="Validation manifest (defaults to --manifest)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=224)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-epochs", type=int, default=100)
    parser.add_argument("--clf-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="AdamW learning rate (paper uses Adam at 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", "--tau", type=float, default=0.1,
                        help="Temperature for the era contrastive loss")
    parser.add_argument("--tsne-max-samples", type=int, default=2000,
                        help="Maximum stratified train+validation examples used for the t-SNE plot")
    parser.add_argument("--tsne-perplexity", type=float, default=30,
                        help="t-SNE perplexity (capped automatically for small datasets)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15,
                        help="UMAP neighborhood size (capped automatically for small datasets)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP minimum distance between points in the 2D layout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.manifest:
        args.train_manifest = args.train_manifest or args.manifest
        args.val_manifest = args.val_manifest or args.manifest
    elif not (args.train_manifest and args.val_manifest):
        default_manifest = Path(args.data_root) / f"dataset_{args.task}" / "manifest.csv"
        if default_manifest.exists():
            args.train_manifest = args.val_manifest = str(default_manifest)
        else:
            parser.error("Provide --manifest, both split manifests, or data/raw/dataset_{task}/manifest.csv")

    if min(args.batch_size, args.backbone_epochs, args.clf_epochs, args.patience, args.proj_dim,
           args.sample_rate, args.n_mels, args.n_fft, args.hop_length) < 1:
        parser.error("batch-size, epochs, patience, projection and audio dimensions must be positive")
    if not 0 < args.seconds <= 30:
        parser.error("seconds must be in (0, 30]")
    if args.lr <= 0 or args.weight_decay < 0 or args.temperature <= 0:
        parser.error("lr and temperature must be positive; weight-decay and beta must be non-negative")
    if args.tsne_max_samples < 3 or args.tsne_perplexity <= 0 or args.umap_n_neighbors < 2:
        parser.error("tsne-max-samples must be at least 3, perplexity positive, and umap-n-neighbors at least 2")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("umap-min-dist must be between 0 and 1")

    seed_everything(args.seed)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(f"Choose a new output directory: {output / 'best.pt'} exists")

    train_rows = read_manifest(args.train_manifest, args.data_root, args.task, "train")
    val_rows = read_manifest(args.val_manifest, args.data_root, args.task, "validation")
    check_disjoint(train_rows, val_rows)
    labels = LABELS[args.task]
    if {row["label"] for row in train_rows} != set(labels):
        raise ValueError("Training split must contain all task labels")
    y_train = torch.tensor([labels.index(row["label"]) for row in train_rows], dtype=torch.long)
    y_val = torch.tensor([labels.index(row["label"]) for row in val_rows], dtype=torch.long)

    ds_train = LabeledAudioDataset(
        AudioDataset(train_rows, sampling_rate=args.sample_rate, seconds=args.seconds), y_train)
    ds_val = LabeledAudioDataset(
        AudioDataset(val_rows, sampling_rate=args.sample_rate, seconds=args.seconds), y_val)
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_audio,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False,
                            collate_fn=collate_audio)

    model_args = {
        "num_classes": len(labels),
        "sample_rate": args.sample_rate,
        "seconds": args.seconds,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "proj_dim": args.proj_dim,
    }
    model = AudioSUCCNNv2(**model_args).to(device)
    criterion_ec = EraContrastiveLoss(temperature=args.temperature)
    backbone_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_json(output / "config.json", vars(args) | {
        "labels": labels,
        "model_architecture": "CNNv2 Audio-SUC",
        "prediction_rule": "classification_head_on_audio_embedding",
    })

    print(f"Training CNNv2 Audio-SUC (tau={args.temperature})...")

    # stage 1: training backbone with ec loss
    for bb_epoch in range(1, args.backbone_epochs + 1):
        model.train()
        loss_ec_sum = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            backbone_opt.zero_grad(set_to_none=True)
            z = model.embed(model.encode(x))
            loss_ec = criterion_ec(z, y)
            loss_ec.backward()
            backbone_opt.step()
            loss_ec_sum += loss_ec.item()
        print(f"Backbone Epoch {bb_epoch}/{args.backbone_epochs}, EC loss: {loss_ec_sum / len(loader):.4f}")

        if bb_epoch % 10 == 0: # stage 2: train clf with ce loss
            stage2_args = {
                "model": model,
                "loader": loader,
                "val_loader": val_loader,
                "output": output,
                "device": device,
                "num_classes": 6,
                "args": args,
                "bb_epoch": bb_epoch,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "model_args": model_args,
            }
            stage2(**stage2_args)


    clf = LinearClassifier(512, 6)
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["backbone_state_dict"], strict=True)
    clf.load_state_dict(best["clf_state_dict"], strict=True)
    all_rows = train_rows + val_rows
    all_labels = torch.cat((y_train, y_val))
    tsne_path, umap_path = save_embedding_visualizations(
        model, all_rows, all_labels, labels, args, device, output)
    print(f"Best checkpoint saved to: {output / 'best.pt'}")
    print(f"t-SNE plot saved to: {tsne_path}")
    print(f"UMAP plot saved to: {umap_path}")


if __name__ == "__main__":
    main()
