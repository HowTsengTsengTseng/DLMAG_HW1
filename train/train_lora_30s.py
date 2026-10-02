"""LoRA SupCon training using two augmented full 30-second views per recording.

This mirrors the stage1 workflow in train_lora.py, with the crop duration fixed
to 30 seconds. Each view is augmented independently in dataset.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence

from augmentations import make_audio_augmentation
from dataset import (
    AudioDataset,
    ClassBalancedBatchSampler,
    FullAudioTwoViewDataset,
    check_disjoint,
    read_manifest,
)
from lora_pipeline import (
    forward_h,
    make_lora_encoder,
    package_versions,
    trainable_report,
)
from models import LinearClassifier, NonLinearClassifier, ProjectionHead
from loss import supcon_loss
from train.train_lora import device_autocast, load_stage1, parameter_groups
from utils import (
    LABELS, MODEL_ID, get_device, metrics, save_confusion, save_json,
    seed_everything,
)
from visualize import add_visualization_args, save_embedding_visualizations


FULL_SECONDS = 30.0


def read_rows(args):
    root = Path(args.data_root)
    manifest = args.manifest or str(root / f"dataset_{args.task}" / "manifest.csv")
    train_rows = read_manifest(args.train_manifest or manifest, args.data_root, args.task, "train")
    val_rows = read_manifest(args.val_manifest or manifest, args.data_root, args.task, "validation")
    check_disjoint(train_rows, val_rows)
    label_map = {label: i for i, label in enumerate(LABELS[args.task])}
    for row in train_rows + val_rows:
        row["label_index"] = label_map[row["label"]]
    return train_rows, val_rows


def save_checkpoint(path, epoch, args, encoder, projection, report):
    from peft import get_peft_model_state_dict

    torch.save({
        "format_version": 2,
        "kind": "mert_lora_stage1",
        "task": args.task,
        "labels": LABELS[args.task],
        "model_id": args.model_id,
        "revision": encoder.revision,
        "adapter_epoch": epoch,
        "adapter_state": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(encoder.backbone).items()},
        "projection_state": {k: v.detach().cpu() for k, v in projection.state_dict().items()},
        "lora": args.lora,
        "crop_seconds": FULL_SECONDS,
        "view_policy": "full_30s_recording_independent_augmentations",
        "pooling": "feature_attention_mask_mean",
        "temperature": args.temperature,
        "seed": args.seed,
        "reports": report,
        "package_versions": package_versions(),
    }, path)


@torch.no_grad()
def visualize_full_recordings(args, encoder, projection, rows, output):
    dataset = AudioDataset(rows, encoder.processor.sampling_rate, seconds=FULL_SECONDS)
    loader = DataLoader(dataset, batch_size=args.extract_batch_size, shuffle=False,
                        collate_fn=lambda batch: pad_sequence(batch, batch_first=True))
    encoder.eval()
    projection.eval()
    h_values, z_values = [], []
    for waveforms in loader:
        h = forward_h(encoder, waveforms)
        h_values.append(h.float().cpu())
        z_values.append(projection(h).float().cpu())
    labels = torch.tensor([row["label_index"] for row in rows], dtype=torch.long)
    paths = save_embedding_visualizations(
        {"h": torch.cat(h_values), "z": torch.cat(z_values)}, labels,
        LABELS[args.task], args, output, prefix="lora_30s_supcon",
        sample_ids=[row["sample_id"] for row in rows],
    )
    print("Saved embedding visualizations: " + ", ".join(str(path) for path in paths.values()))


@torch.no_grad()
def extract_full_embeddings(rows, encoder, batch_size, device):
    """Extract one h vector per complete recording, padding only within batches."""
    dataset = AudioDataset(rows, encoder.processor.sampling_rate, seconds=FULL_SECONDS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: pad_sequence(batch, batch_first=True))
    values = [forward_h(encoder, waveforms.to(device)).float().cpu()
              for waveforms in loader]
    return torch.cat(values)


def classifier_type(name):
    return LinearClassifier if name == "linear" else NonLinearClassifier


def probe_checkpoint(args, checkpoint_path, output, global_best_key, history):
    seed_everything(args.seed)
    device = get_device(args.device)
    train_rows, val_rows = read_rows(args)
    stage1_ckpt, encoder, _ = load_stage1(checkpoint_path, device)
    h_train = extract_full_embeddings(train_rows, encoder, args.extract_batch_size, device)
    h_val = extract_full_embeddings(val_rows, encoder, args.extract_batch_size, device)
    y_train = torch.tensor([row["label_index"] for row in train_rows], dtype=torch.long)
    y_val = torch.tensor([row["label_index"] for row in val_rows], dtype=torch.long)

    mean = h_train.mean(0)
    std = h_train.std(0, unbiased=False).clamp_min(args.std_floor)
    x_train = (h_train - mean) / std
    x_val = (h_val - mean) / std
    model = classifier_type(args.classifier)(x_train.shape[1], len(LABELS[args.task]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr,
        weight_decay=args.classifier_weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=args.classifier_batch_size, shuffle=True)
    local_best_key = (-1.0, float("-inf"))
    stale = 0

    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            loss = F.cross_entropy(logits, y_val).item()
        score = metrics(logits, y_val, LABELS[args.task]) | {"loss": loss}
        key = (score["top1"], -score["loss"])
        entry = {"stage1_checkpoint": str(checkpoint_path), "epoch": epoch, **score}
        history.append(entry)
        print(f"{checkpoint_path.name} probe_epoch={epoch} {score}", flush=True)

        if key > local_best_key:
            local_best_key, stale = key, 0
        else:
            stale += 1

        if key > global_best_key:
            global_best_key = key
            checkpoint = {
                "format_version": 2,
                "kind": "mert_lora_probe_30s",
                "task": args.task,
                "labels": LABELS[args.task],
                "model_id": stage1_ckpt["model_id"],
                "revision": stage1_ckpt["revision"],
                "lora": stage1_ckpt["lora"],
                "adapter_state": stage1_ckpt["adapter_state"],
                "projection_state": stage1_ckpt["projection_state"],
                "adapter_epoch": stage1_ckpt["adapter_epoch"],
                "stage1_checkpoint": str(checkpoint_path),
                "classifier": args.classifier,
                "classifier_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_args": {"input_dim": x_train.shape[1], "num_classes": len(LABELS[args.task])},
                "feature_mean": mean,
                "feature_std": std,
                "seconds": FULL_SECONDS,
                "pooling": "feature_attention_mask_mean",
                "metrics": score,
                "epoch": epoch,
                "seed": args.seed,
                "train_ids": [row["sample_id"] for row in train_rows],
                "validation_ids": [row["sample_id"] for row in val_rows],
                "package_versions": package_versions(),
            }
            torch.save(checkpoint, output / "best.pt")
            save_json(output / "validation_metrics.json", entry)
            save_confusion(score, output / "validation_confusion.png")
            save_json(output / "history.json", history)

        if stale >= args.classifier_patience:
            break

    return global_best_key


def run_probe(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "probe_config.json", vars(args) | {
        "seconds": FULL_SECONDS,
        "pooling": "feature_attention_mask_mean",
        "labels": LABELS[args.task],
    })
    source = Path(args.stage1)
    if args.probe_all:
        checkpoint_paths = sorted(source.glob("epoch_*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No epoch_*.pt stage1 checkpoints found in {source}")
    else:
        if not source.is_file():
            raise FileNotFoundError(f"Stage1 checkpoint not found: {source}")
        checkpoint_paths = [source]

    history = []
    best_key = (-1.0, float("-inf"))
    for checkpoint_path in checkpoint_paths:
        print(f"Probing full-30s checkpoint: {checkpoint_path}", flush=True)
        best_key = probe_checkpoint(args, checkpoint_path, output, best_key, history)
    save_json(output / "history.json", history)

    best_path = output / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Probe finished without a best checkpoint")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    print(f"Best probe checkpoint: {best_path} "
          f"(stage1={best['stage1_checkpoint']}, metrics={best['metrics']})", flush=True)
    if args.visualize:
        device = get_device(args.device)
        _, encoder, _ = load_stage1(best_path, device)
        projection = ProjectionHead(encoder.hidden_size).to(device)
        projection.load_state_dict(best["projection_state"])
        train_rows, val_rows = read_rows(args)
        visualize_full_recordings(args, encoder, projection,
                                  train_rows + val_rows, output)


def train(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("epoch_*.pt")):
        raise FileExistsError(f"Choose a new output directory; stage1 checkpoints already exist in {output}")

    train_rows, val_rows = read_rows(args)
    if {row["label"] for row in train_rows} != set(LABELS[args.task]):
        raise ValueError("Training split must contain every task label for class-balanced SupCon batches")
    encoder, report = make_lora_encoder(args.model_id, args.revision, device, args.lora)
    projection = ProjectionHead(encoder.hidden_size).to(device)
    report["trainable"] = trainable_report(encoder)
    save_json(output / "model_inspection.json", report)

    dataset = FullAudioTwoViewDataset(
        train_rows, encoder.processor.sampling_rate, seconds=FULL_SECONDS,
        transform=make_audio_augmentation(encoder.processor.sampling_rate, FULL_SECONDS))
    sampler = ClassBalancedBatchSampler(
        train_rows, args.classes_per_batch, args.recordings_per_class, args.batches_per_epoch)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    groups = parameter_groups(encoder.named_parameters(), args.adapter_lr, args.weight_decay)
    groups += parameter_groups(projection.named_parameters(), args.projection_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(groups)
    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = max(1, round(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) /
                                   max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    save_json(output / "config.json", vars(args) | {
        "seconds": FULL_SECONDS,
        "view_policy": "full_30s_recording_independent_augmentations",
        "revision": encoder.revision,
        "labels": LABELS[args.task],
        "package_versions": package_versions(),
    })

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projection.train()
        loss_sum = 0.0
        for view1, view2, labels in loader:
            waves = torch.cat((view1, view2), dim=0).to(device)
            labels = labels.to(device)
            paired_labels = torch.cat((labels, labels))
            optimizer.zero_grad(set_to_none=True)
            with device_autocast(device):
                h = forward_h(encoder, waves)
                z = projection(h)
            loss = supcon_loss(z, paired_labels, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projection.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()

        print(f"epoch={epoch} supcon={loss_sum / len(loader):.5f}", flush=True)
        if epoch % 5 == 0:
            save_checkpoint(output / f"epoch_{epoch:03d}.pt", epoch, args,
                            encoder, projection, report)

    final_checkpoint = output / f"epoch_{args.epochs:03d}.pt"
    if args.epochs % 5:
        save_checkpoint(final_checkpoint, args.epochs, args, encoder, projection, report)
    print(f"Final checkpoint saved to: {final_checkpoint}")

    if args.visualize:
        visualize_full_recordings(args, encoder, projection, train_rows + val_rows, output)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["stage1", "probe"], default="stage1")
    parser.add_argument("--task", choices=LABELS, required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--classes-per-batch", type=int, default=6)
    parser.add_argument("--recordings-per-class", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--projection-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lora", type=json.loads, default={"r": 8, "alpha": 16, "dropout": 0.05})
    parser.add_argument("--stage1", help="Stage1 checkpoint file, or directory when --probe-all is used")
    parser.add_argument("--probe-all", action="store_true",
                        help="Probe all epoch_*.pt checkpoints in --stage1 and select the global best")
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--classifier", choices=["linear", "nonlinear"], default="linear")
    parser.add_argument("--classifier-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=15)
    add_visualization_args(parser)
    args = parser.parse_args()

    if args.visualize_max_samples < 3 or args.tsne_perplexity <= 0 or args.umap_n_neighbors < 2:
        parser.error("invalid visualization sample count, perplexity, or UMAP neighbor count")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("umap-min-dist must be between 0 and 1")
    if args.mode == "stage1":
        if min(args.extract_batch_size, args.classes_per_batch,
               args.recordings_per_class, args.epochs) < 1:
            parser.error("batch size, classes per batch, recordings per class, and epochs must be positive")
        if args.temperature <= 0 or args.adapter_lr <= 0 or args.projection_lr <= 0 or args.weight_decay < 0:
            parser.error("temperature and learning rates must be positive; weight decay must be non-negative")
        if args.classes_per_batch != len(LABELS[args.task]):
            parser.error(f"classes-per-batch must be {len(LABELS[args.task])} for task {args.task}")
        train(args)
    else:
        if not args.stage1:
            parser.error("probe mode requires --stage1")
        if min(args.extract_batch_size, args.classifier_batch_size,
               args.classifier_epochs, args.classifier_patience) < 1:
            parser.error("probe batch size, epochs, and patience must be positive")
        if args.classifier_lr <= 0 or args.classifier_weight_decay < 0 or args.std_floor <= 0:
            parser.error("classifier lr and std-floor must be positive; weight decay must be non-negative")
        run_probe(args)


if __name__ == "__main__":
    main()
