"""Compute validation loss, top-1 accuracy, and top-3 accuracy.

Task A and Task B can use different checkpoint families. Crop-based
checkpoints are evaluated after their crop predictions are combined at the
recording level, matching inference behavior.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv
import torch
from torch.nn import functional as F

from common import (
    add_prediction_args,
    validate_checkpoint_task,
)
from dataset import read_manifest
from predict import CONTRASTIVE_KINDS, LORA_KINDS, predict_lora, predict_mert
from utils import LABELS, get_device, save_json


def load_validation_rows(data_root, task, manifest=None):
    if manifest is None:
        candidates = [
            Path(data_root) / f"dataset_{task}" / "manifest.csv",
            Path("data/manifests") / f"{task}_validation.csv",
            Path("data/manifests") / f"{task}_val.csv",
        ]
        manifest_path = next((path for path in candidates if path.exists()), None)
        if manifest_path is None:
            raise FileNotFoundError(
                f"Cannot find a validation manifest for task {task}; "
                f"pass --manifest-{task.lower()}")
    else:
        manifest_path = Path(manifest)
    return read_manifest(manifest_path, data_root, task, "validation")


def score_task(args, task, checkpoint_path, manifest, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_task(checkpoint, task, checkpoint_path)
    rows = load_validation_rows(args.data_root, task, manifest)
    kind = checkpoint.get("kind")

    if kind in LORA_KINDS:
        logits, labels = predict_lora(checkpoint, rows, args, device)
    elif kind in CONTRASTIVE_KINDS or kind is None:
        logits, labels = predict_mert(checkpoint, rows, args, device)
    else:
        raise ValueError(f"Unsupported checkpoint kind {kind!r}: {checkpoint_path}")

    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = torch.tensor(
        [label_to_index[row["label"]] for row in rows], dtype=torch.long)
    loss = F.cross_entropy(logits, targets).item()
    ranked = logits.argsort(dim=1, descending=True)
    top1 = (ranked[:, 0] == targets).float().mean().item()
    top3 = (ranked[:, :3] == targets[:, None]).any(dim=1).float().mean().item()
    return {
        "checkpoint": str(checkpoint_path),
        "kind": kind or "mert_mlp",
        "n_samples": len(rows),
        "loss": loss,
        "top1": top1,
        "top3": top3,
        "labels": labels,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    add_prediction_args(parser)
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--extract-batch-size", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.extract_batch_size) < 1:
        parser.error("batch sizes must be positive")

    device = get_device(args.device)
    results = {}
    for task, checkpoint_path, manifest in (
        ("A", args.checkpoint_a, args.manifest_a),
        ("B", args.checkpoint_b, args.manifest_b),
    ):
        results[f"dataset_{task}"] = score_task(
            args, task, checkpoint_path, manifest, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(args.output, results)
    for task, score in results.items():
        print(
            f"{task}: loss={score['loss']:.6f} "
            f"top1={score['top1']:.4f} top3={score['top3']:.4f} "
            f"n={score['n_samples']}",
            flush=True,
        )
    print(f"Saved validation metrics to {args.output}")


if __name__ == "__main__":
    main()
