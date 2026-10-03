"""Train a nonlinear classifier on frozen stage1 projection embeddings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from lora_pipeline import fixed_crops, package_versions, save_json
from models import NonLinearClassifier, ProjectionHead
from train.lora_checkpoint_utils import load_stage1, pooled_crops, rows_for
from utils import (LABELS, combine_and_record_scores, get_device, metrics,
                   save_confusion, seed_everything)


@torch.no_grad()
def extract_z(encoder, projection, rows, crop_seconds, device, batch_size):
    crops = fixed_crops(rows, encoder.processor.sampling_rate, crop_seconds)
    h = pooled_crops(encoder, crops, device, batch_size)
    return torch.cat([
        projection(batch.to(device)).float().cpu()
        for batch in h.split(batch_size)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=LABELS, required=True)
    parser.add_argument("--stage1", required=True, help="Path to an epoch_*.pt stage1 checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=15)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--crop-seconds", type=float, default=None,
                        help="Defaults to the crop duration stored in the checkpoint")
    args = parser.parse_args()
    if args.extract_batch_size < 1 or args.classifier_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.classifier_epochs < 1 or args.classifier_patience < 1:
        parser.error("classifier epochs and patience must be positive")

    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt, encoder = load_stage1(args.stage1, device)
    crop_seconds = (args.crop_seconds if args.crop_seconds is not None
                    else ckpt.get("crop_seconds", 10))
    if not 0 < crop_seconds <= 30:
        parser.error("crop duration must be in (0, 30]")
    train_rows, val_rows, _ = rows_for(args)
    projection = ProjectionHead(encoder.hidden_size).to(device)
    projection.load_state_dict(ckpt["projection_state"])
    encoder.eval()
    projection.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in projection.parameters():
        parameter.requires_grad_(False)

    # Cache frozen z once; only the nonlinear classifier is optimized below.
    z_train = extract_z(encoder, projection, train_rows, crop_seconds, device,
                        args.extract_batch_size)
    z_val = extract_z(encoder, projection, val_rows, crop_seconds, device,
                      args.extract_batch_size)
    y_train = torch.tensor([row["label_index"] for row in train_rows]).repeat_interleave(3)
    y_val = torch.tensor([row["label_index"] for row in val_rows]).repeat_interleave(3)
    mean = z_train.mean(0)
    std = z_train.std(0, unbiased=False).clamp_min(args.std_floor)
    z_train = (z_train - mean) / std
    z_val = (z_val - mean) / std

    classifier = NonLinearClassifier(z_train.shape[1], len(LABELS[args.task])).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.classifier_lr,
                                  weight_decay=args.classifier_weight_decay)
    loader = DataLoader(TensorDataset(z_train, y_train),
                        batch_size=args.classifier_batch_size, shuffle=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_key = (-1.0, float("inf"))
    best_metrics = None
    stale = 0
    for epoch in range(1, args.classifier_epochs + 1):
        classifier.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(classifier(features), targets).backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            logits = classifier(z_val.to(device)).float().cpu()
        probabilities, score = combine_and_record_scores(logits, y_val, len(val_rows))
        print(f"epoch={epoch} {score}", flush=True)
        key = (score["top1"], -score["loss"])
        if key > best_key:
            best_key = key
            best_metrics = score
            stale = 0
            checkpoint = {
                "format_version": 1,
                "kind": "mert_lora_projection_nonlinear_probe",
                "task": args.task,
                "labels": LABELS[args.task],
                "model_id": ckpt["model_id"],
                "revision": ckpt["revision"],
                "adapter_state": ckpt["adapter_state"],
                "adapter_epoch": ckpt["adapter_epoch"],
                "stage1_checkpoint": str(Path(args.stage1).resolve()),
                "projection_state": ckpt["projection_state"],
                "lora": ckpt["lora"],
                "crop_seconds": crop_seconds,
                "pooling": ckpt.get("pooling", "feature_attention_mask_mean"),
                "classifier_state": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
                "classifier_args": {"input_dim": z_train.shape[1],
                                    "num_classes": len(LABELS[args.task])},
                "feature_mean": mean,
                "feature_std": std,
                "metrics": score,
                "train_ids": [row["sample_id"] for row in train_rows],
                "validation_ids": [row["sample_id"] for row in val_rows],
                "package_versions": package_versions(),
            }
            torch.save(checkpoint, out / "best.pt")
            save_json(out / "validation_metrics.json", score | {"epoch": epoch})
            record_targets = torch.tensor([row["label_index"] for row in val_rows])
            save_confusion(metrics(probabilities.clamp_min(1e-8).log(), record_targets,
                                   LABELS[args.task]), out / "validation_confusion.png")
        else:
            stale += 1
        if stale >= args.classifier_patience:
            break

    print(f"Best validation metrics: {best_metrics}", flush=True)
    print(f"Saved classifier checkpoint: {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
