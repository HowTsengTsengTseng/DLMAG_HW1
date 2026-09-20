"""Inference entry point (not a unit test): export both official test splits as top-3 JSON."""
import argparse

from dotenv import load_dotenv
import torch

from dataset import read_manifest
from features import extract_features
from models import MERTEncoder, MLPClassifier
from utils import LABELS, get_device, save_json


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw", help="Audio root on the grader's device (default: data/raw)")
    parser.add_argument("--manifest-a", help="Manifest for task A (defaults to data/raw/dataset_A/manifest.csv or data/manifests/A_test.csv)")
    parser.add_argument("--manifest-b", help="Manifest for task B (defaults to data/raw/dataset_B/manifest.csv or data/manifests/B_test.csv)")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output", required=True, help="e.g. predictions/b13902135.json")
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--template", help="Optional official format JSON; validate sample IDs only, never use example answers")
    args = parser.parse_args()
    if not args.manifest_a:
        from pathlib import Path
        for p in ["data/manifests/A_test.csv", f"{args.data_root}/dataset_A/manifest.csv"]:
            if Path(p).exists():
                args.manifest_a = p
                break
        if not args.manifest_a:
            parser.error("Cannot find default manifest for task A; please provide --manifest-a")
    if not args.manifest_b:
        from pathlib import Path
        for p in ["data/manifests/B_test.csv", f"{args.data_root}/dataset_B/manifest.csv"]:
            if Path(p).exists():
                args.manifest_b = p
                break
        if not args.manifest_b:
            parser.error("Cannot find default manifest for task B; please provide --manifest-b")
    device = get_device(args.device)
    predictions = {}
    encoder, encoder_key = None, None
    for task, manifest, checkpoint_path in [
        ("A", args.manifest_a, args.checkpoint_a), ("B", args.manifest_b, args.checkpoint_b)
    ]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1 or checkpoint["task"] != task or checkpoint["labels"] != LABELS[task]:
            raise ValueError(f"Wrong checkpoint task/labels/version: {checkpoint_path}")
        rows = read_manifest(manifest, args.data_root, task, "test")
        fitted_ids = set(checkpoint["train_ids"]) | set(checkpoint["validation_ids"])
        if fitted_ids & {row["sample_id"] for row in rows}:
            raise ValueError("Test IDs overlap with training/validation IDs")
        key = (checkpoint["model_id"], checkpoint["revision"])
        if key != encoder_key:
            if encoder is not None:
                del encoder
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            encoder = MERTEncoder(*key).to(device)
            encoder_key = key
        x = extract_features(rows, encoder, args.cache_dir, checkpoint["model_id"],
                             checkpoint["seconds"], args.extract_batch_size)
        model = MLPClassifier(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        with torch.no_grad():
            ranks = model(x.to(device)).argsort(dim=1, descending=True)[:, :3].cpu().tolist()
        labels = checkpoint["labels"]
        predictions[f"dataset_{task}"] = {
            row["sample_id"]: [labels[i] for i in rank] for row, rank in zip(rows, ranks)
        }
    if args.template:
        import json
        with open(args.template, encoding="utf-8") as f:
            template = json.load(f)
        for task in ("dataset_A", "dataset_B"):
            if set(predictions[task]) != set(template[task]):
                raise ValueError(f"Missing or extra sample IDs relative to official template: {task}")
    save_json(args.output, predictions)
    print(f"Saved {args.output}: " + ", ".join(f"{task}={len(rows)}" for task, rows in predictions.items()))


if __name__ == "__main__":
    main()
