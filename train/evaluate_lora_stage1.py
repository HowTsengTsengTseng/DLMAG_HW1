"""Evaluate a LoRA stage1 checkpoint with prototype-based validation.

The checkpoint is evaluated without training a classifier. Class prototypes
are computed from normalized training embeddings produced by the stage1
projection head, and validation logits are cosine similarities to those
prototypes divided by the saved contrastive temperature.

Both crop-level and recording-level metrics are reported. Crop-level metrics
match the validation calculation used by train/train_contrastive.py. Recording-
level metrics average the three crop probabilities before computing the loss,
matching the existing fixed-crop inference rule used by the LoRA workflow.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.nn import functional as F
from dotenv import load_dotenv

from train.train_lora import load_stage1, pooled_crops, rows_for
from lora_pipeline import fixed_crops
from utils import LABELS, get_device, metrics, save_confusion, save_json


@torch.no_grad()
def projection_embeddings(encoder, projection, waves, device, batch_size):
    """Extract normalized projection embeddings while keeping results on CPU."""
    h = pooled_crops(encoder, waves, device, batch_size)
    projection.eval()
    values = [projection(batch.to(device)).float().cpu()
              for batch in h.split(batch_size)]
    return torch.cat(values)


@torch.no_grad()
def compute_class_prototypes(embeddings, labels, num_classes):
    """Compute a normalized arithmetic-mean embedding for each class."""
    prototypes = torch.zeros(
        (num_classes, embeddings.shape[1]),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    for class_index in range(num_classes):
        class_embeddings = embeddings[labels == class_index]
        if not class_embeddings.numel():
            raise ValueError(
                f"Training split has no embeddings for class {class_index}"
            )
        prototypes[class_index] = F.normalize(class_embeddings.mean(0), dim=-1)
    return prototypes


def recording_scores(crop_logits, crop_labels, num_recordings, labels):
    """Aggregate three crop probabilities and calculate recording metrics."""
    crop_probs = crop_logits.softmax(dim=-1).reshape(num_recordings, 3, -1)
    recording_probs = crop_probs.mean(dim=1)
    recording_labels = crop_labels.reshape(num_recordings, 3)[:, 0]
    recording_loss = (
        -recording_probs[
            torch.arange(num_recordings), recording_labels
        ].clamp_min(1e-8).log()
    ).mean().item()
    recording_logits = recording_probs.clamp_min(1e-8).log()
    score = metrics(recording_logits, recording_labels, labels)
    score["loss"] = recording_loss
    return score


def evaluate(args):
    load_dotenv()
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # The checkpoint is authoritative for the model and LoRA configuration.
    ckpt, encoder, _ = load_stage1(args.stage1, device)
    train_rows, val_rows, _ = rows_for(args)
    crop_seconds = (args.crop_seconds if args.crop_seconds is not None
                    else ckpt.get("crop_seconds", 10.0))
    temperature = (args.temperature if args.temperature is not None
                   else ckpt.get("temperature", 0.1))
    if crop_seconds <= 0 or temperature <= 0:
        raise ValueError("crop-seconds and temperature must be positive")

    train_crops = fixed_crops(
        train_rows, encoder.processor.sampling_rate, crop_seconds)
    val_crops = fixed_crops(
        val_rows, encoder.processor.sampling_rate, crop_seconds)
    train_labels = torch.tensor(
        [row["label_index"] for row in train_rows], dtype=torch.long
    ).repeat_interleave(3)
    val_labels = torch.tensor(
        [row["label_index"] for row in val_rows], dtype=torch.long
    ).repeat_interleave(3)

    encoder.eval()
    from models import ProjectionHead
    projection = ProjectionHead(encoder.hidden_size).to(device)
    projection.load_state_dict(ckpt["projection_state"], strict=True)
    projection.eval()

    train_z = projection_embeddings(
        encoder, projection, train_crops, device, args.extract_batch_size)
    val_z = projection_embeddings(
        encoder, projection, val_crops, device, args.extract_batch_size)
    prototypes = compute_class_prototypes(
        train_z, train_labels, len(LABELS[args.task]))
    crop_logits = (val_z @ prototypes.T / temperature).float()

    crop_loss = F.cross_entropy(crop_logits, val_labels).item()
    if not math.isfinite(crop_loss):
        raise RuntimeError("Non-finite crop-level validation loss")
    crop_scores = metrics(crop_logits, val_labels, LABELS[args.task])
    crop_scores["loss"] = crop_loss
    record_scores = recording_scores(
        crop_logits, val_labels, len(val_rows), LABELS[args.task])

    result = {
        "checkpoint": str(Path(args.stage1)),
        "task": args.task,
        "temperature": temperature,
        "crop_seconds": crop_seconds,
        "num_train_recordings": len(train_rows),
        "num_validation_recordings": len(val_rows),
        "num_classes": len(LABELS[args.task]),
        "crop_level": crop_scores,
        "recording_level": record_scores,
    }
    save_json(output / "validation_metrics.json", result)
    save_confusion(record_scores, output / "validation_confusion.png")
    print(
        f"Crop level: loss={crop_loss:.5f} "
        f"top1={crop_scores['top1'] * 100:.2f}% "
        f"top3={crop_scores['top3'] * 100:.2f}%"
    )
    print(
        f"Recording level: loss={record_scores['loss']:.5f} "
        f"top1={record_scores['top1'] * 100:.2f}% "
        f"top3={record_scores['top3'] * 100:.2f}%"
    )
    print(f"Saved metrics to {output / 'validation_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=LABELS)
    parser.add_argument("--stage1", required=True,
                        help="Path to a stage1 checkpoint (*.pt)")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--crop-seconds", type=float,
                        help="Override checkpoint crop duration")
    parser.add_argument("--temperature", type=float,
                        help="Override checkpoint contrastive temperature")
    args = parser.parse_args()
    if args.extract_batch_size < 1:
        parser.error("extract-batch-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
