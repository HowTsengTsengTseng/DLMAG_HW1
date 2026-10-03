"""Visualize h and z embeddings from a selected MERT LoRA stage1 checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lora_pipeline import fixed_crops
from models import ProjectionHead
from train.lora_checkpoint_utils import load_stage1, pooled_crops, rows_for
from utils import LABELS, get_device, seed_everything
from visualize import save_embedding_visualizations


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
    parser.add_argument("--crop-seconds", type=float, default=None,
                        help="Defaults to the crop duration stored in the checkpoint")
    parser.add_argument("--visualize-max-samples", "--tsne-max-samples",
                        dest="visualize_max_samples", type=int, default=2000)
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

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

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        crops = fixed_crops(rows, encoder.processor.sampling_rate, crop_seconds)
        h = pooled_crops(encoder, crops, device, args.extract_batch_size)
        with torch.no_grad():
            z = torch.cat([
                projection(batch.to(device)).float().cpu()
                for batch in h.split(args.extract_batch_size)
            ])
        targets = torch.tensor([row["label_index"] for row in rows]).repeat_interleave(3)
        crop_ids = [f"{row['sample_id']}#crop{crop}" for row in rows for crop in range(3)]
        paths = save_embedding_visualizations(
            {"h": h, "z": z}, targets, LABELS[args.task], args, output,
            prefix=f"lora_stage1_{split_name}", sample_ids=crop_ids)
        print(f"{split_name}: saved " + ", ".join(str(path) for path in paths.values()), flush=True)


if __name__ == "__main__":
    main()
