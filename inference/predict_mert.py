"""Predict test top-3 labels from train.py or train_contrastive.py checkpoints."""
import argparse
from pathlib import Path

from dotenv import load_dotenv
import torch

from common import (
    add_prediction_args,
    load_test_rows,
    rank_predictions,
    validate_checkpoint_task,
    validate_no_training_overlap,
    write_predictions,
)
from features import extract_features
from models import MERTEncoder, MLPClassifier
from utils import LABELS, get_device


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
    predictions = {}
    encoder = None
    encoder_key = None
    for task, manifest, checkpoint_path in (
        ("A", args.manifest_a, args.checkpoint_a),
        ("B", args.manifest_b, args.checkpoint_b),
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint_task(checkpoint, task, checkpoint_path)
        if checkpoint.get("format_version") != 1 or "state_dict" not in checkpoint:
            raise ValueError(f"Expected a train.py/train_contrastive.py MLP checkpoint: {checkpoint_path}")
        rows = load_test_rows(args.data_root, task, manifest)
        validate_no_training_overlap(checkpoint, rows, checkpoint_path)

        key = (checkpoint["model_id"], checkpoint["revision"])
        if key != encoder_key:
            if encoder is not None:
                del encoder
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            encoder = MERTEncoder(*key).to(device).eval()
            encoder_key = key

        features = extract_features(
            rows, encoder, args.cache_dir, checkpoint["model_id"],
            checkpoint["seconds"], args.extract_batch_size)
        model = MLPClassifier(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        with torch.no_grad():
            logits = torch.cat([
                model(batch.to(device)).cpu()
                for batch in features.split(args.batch_size)
            ])
        top3 = rank_predictions(logits, checkpoint["labels"])
        predictions[f"dataset_{task}"] = {
            row["sample_id"]: ranked for row, ranked in zip(rows, top3)
        }

    write_predictions(args.output, predictions, args.template)


if __name__ == "__main__":
    main()
