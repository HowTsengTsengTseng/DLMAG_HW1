"""Two-stage contrastive training with random stage1 crops and crop aggregation.

This is the crop counterpart of train/train_contrastive.py. It keeps the
full-audio workflow intact: stage1 trains the projection with two fresh random
crops per recording and EC loss, while probe uses three deterministic crops
and combines their probabilities at recording level.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import ClassBalancedBatchSampler, TwoViewDataset
from lora_pipeline import fixed_crops, forward_h
from loss import EraContrastiveLoss
from models import MERTEncoder, MLPClassifier, NonLinearClassifier
from train.train_contrastive import (
    cpu_state_dict, load_stage1_checkpoint, read_probe_rows, read_train_rows,
    resolve_manifests,
)
from utils import (
    LABELS, MODEL_ID, combine_and_record_scores, get_device, metrics,
    save_confusion, save_json, seed_everything,
)
from visualize import add_visualization_args, save_embedding_visualizations


def extract_crop_features(rows, model_id, revision, crop_seconds, device,
                          batch_size, keep_encoder=False):
    encoder = MERTEncoder(model_id, revision).to(device).eval()
    resolved_revision = encoder.revision
    input_dim = encoder.hidden_size
    crops = fixed_crops(rows, encoder.processor.sampling_rate, crop_seconds)
    features = torch.cat([
        forward_h(encoder, batch.to(device)).float().cpu()
        for batch in crops.split(batch_size)
    ])
    if keep_encoder:
        return features, resolved_revision, input_dim, encoder
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, resolved_revision, input_dim


@torch.no_grad()
def encode_batches(projection, features, device, batch_size):
    projection.eval()
    return torch.cat([
        projection.encode(batch.to(device)).float().cpu()
        for batch in features.split(batch_size)
    ])


def save_stage1_checkpoint(path, epoch, args, projection, projection_args,
                           revision, train_rows, train_loss):
    torch.save({
        "format_version": 2,
        "kind": "mert_contrastive_stage1_crops",
        "task": args.task,
        "labels": LABELS[args.task],
        "model_id": args.model_id,
        "revision": revision,
        "seconds": 30.0,
        "crop_seconds": args.crop_seconds,
        "num_crops": 3,
        "model_args": projection_args,
        "state_dict": cpu_state_dict(projection),
        "epoch": epoch,
        "seed": args.seed,
        "train_ids": [row["sample_id"] for row in train_rows],
        "training_loss": "era_contrastive_loss",
        "train_ec_loss": train_loss,
        "temperature": args.temperature,
    }, path)


def run_stage1(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("epoch_*.pt")):
        raise FileExistsError(
            f"Choose a new output directory; stage1 checkpoints exist in {output}")

    train_rows = read_train_rows(args)
    labels = LABELS[args.task]
    label_map = {label: index for index, label in enumerate(labels)}
    for row in train_rows:
        row["label_index"] = label_map[row["label"]]
    # Fixed crops are used only to fit the training feature standardizer and
    # for optional visualization. Stage1 itself learns from fresh random views.
    x_train_stats, revision, input_dim, encoder = extract_crop_features(
        train_rows, args.model_id, args.revision, args.crop_seconds,
        device, args.extract_batch_size, keep_encoder=True)
    y_train = torch.tensor(
        [labels.index(row["label"]) for row in train_rows], dtype=torch.long
    )

    projection_args = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": len(labels),
        "dropout": args.dropout,
    }
    projection = MLPClassifier(**projection_args)
    projection.fit_standardizer(x_train_stats)
    projection.to(device)
    criterion = EraContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)
    dataset = TwoViewDataset(
        train_rows, sample_rate=encoder.processor.sampling_rate,
        crop_seconds=args.crop_seconds)
    sampler = ClassBalancedBatchSampler(
        train_rows, classes=len(labels),
        recordings_per_class=args.recordings_per_class,
        batches=args.batches_per_epoch)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    save_json(output / "config.json", vars(args) | {
        "resolved_revision": revision,
        "labels": labels,
        "objective": "stage1_era_contrastive_loss",
        "crop_policy": "random_two_views_for_stage1",
    })
    history = []
    for epoch in range(1, args.epochs + 1):
        projection.train()
        total_loss = 0.0
        total_examples = 0
        for view1, view2, y in loader:
            waves = torch.cat([view1, view2], dim=0).to(device)
            y = y.to(device)
            labels2 = torch.cat([y, y])
            optimizer.zero_grad(set_to_none=True)
            h = forward_h(encoder, waves)
            loss = criterion(projection.encode(h), labels2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_examples += len(y)

        train_loss = total_loss / max(1, total_examples)
        history.append({"epoch": epoch, "train_ec_loss": train_loss})
        save_json(output / "stage1_history.json", history)
        message = f"Stage1 epoch {epoch:03d} | EC loss: {train_loss:.5f}"
        if epoch % 5 == 0:
            checkpoint_path = output / f"epoch_{epoch:03d}.pt"
            save_stage1_checkpoint(
                checkpoint_path, epoch, args, projection, projection_args,
                revision, train_rows, train_loss)
            message += f" | saved: {checkpoint_path}"
        print(message, flush=True)

    if args.epochs % 5:
        checkpoint_path = output / f"epoch_{args.epochs:03d}.pt"
        save_stage1_checkpoint(
            checkpoint_path, args.epochs, args, projection, projection_args,
            revision, train_rows, history[-1]["train_ec_loss"])
        print(f"Saved final stage1 checkpoint: {checkpoint_path}", flush=True)

    if args.visualize:
        x_train = x_train_stats
        y_visualize = y_train.repeat_interleave(3)
        z_train = encode_batches(
            projection, x_train, device, args.extract_batch_size)
        paths = save_embedding_visualizations(
            {"h": x_train, "z": z_train}, y_visualize, labels, args, output,
            prefix="contrastive_crops_stage1",
            sample_ids=[f"{row['sample_id']}#crop{crop}"
                        for row in train_rows for crop in range(3)])
        print("Saved embedding visualizations: " +
              ", ".join(str(path) for path in paths.values()))


def combine_recording_predictions(logits, crop_labels, num_recordings, labels):
    probabilities, _ = combine_and_record_scores(
        logits, crop_labels, num_recordings)
    recording_labels = crop_labels[::3][:num_recordings]
    score = metrics(
        probabilities.clamp_min(1e-8).log(), recording_labels, labels)
    score["loss"] = (
        -probabilities[
            torch.arange(num_recordings), recording_labels
        ].clamp_min(1e-8).log()
    ).mean().item()
    return probabilities, score


def probe_checkpoint(args, checkpoint_path, x_train, y_train, x_val, y_val,
                     train_rows, val_rows, output, global_best, history,
                     device):
    seed_everything(args.seed)
    checkpoint = load_stage1_checkpoint(checkpoint_path, args.task)
    if checkpoint.get("kind") != "mert_contrastive_stage1_crops":
        raise ValueError(f"Expected a crop stage1 checkpoint: {checkpoint_path}")
    projection = MLPClassifier(**checkpoint["model_args"])
    projection.load_state_dict(checkpoint["state_dict"], strict=True)
    projection.to(device).eval()
    z_train = encode_batches(projection, x_train, device, args.extract_batch_size)
    z_val = encode_batches(projection, x_val, device, args.extract_batch_size)

    classifier_args = {
        "dim": z_train.shape[1],
        "classes": len(LABELS[args.task]),
        "hidden_dim": args.classifier_hidden_dim,
    }
    classifier = NonLinearClassifier(**classifier_args).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.classifier_lr,
        weight_decay=args.classifier_weight_decay)
    loader = DataLoader(
        TensorDataset(z_train, y_train),
        batch_size=args.classifier_batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))

    checkpoint_best = (-1.0, float("inf"))
    stale = 0
    for epoch in range(1, args.classifier_epochs + 1):
        classifier.train()
        total_loss = 0.0
        for z, y in loader:
            z, y = z.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(z), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        classifier.eval()
        with torch.no_grad():
            crop_logits = classifier(z_val.to(device)).cpu()
        _, score = combine_recording_predictions(
            crop_logits, y_val, len(val_rows), LABELS[args.task])
        key = (score["top1"], -score["loss"])
        entry = {
            "stage1_checkpoint": str(checkpoint_path),
            "stage1_epoch": checkpoint["epoch"],
            "probe_epoch": epoch,
            "train_ce_loss": total_loss / len(y_train),
            **score,
        }
        history.append(entry)
        print(
            f"{checkpoint_path.name} probe_epoch={epoch:03d} "
            f"train_ce={entry['train_ce_loss']:.5f} "
            f"record_val_ce={score['loss']:.5f} "
            f"top1={score['top1'] * 100:.2f}% "
            f"top3={score['top3'] * 100:.2f}%", flush=True)

        if key > (checkpoint_best[0], -checkpoint_best[1]):
            checkpoint_best = (score["top1"], score["loss"])
            stale = 0
        else:
            stale += 1

        if key > (global_best[0], -global_best[1]):
            global_best = (score["top1"], score["loss"])
            final_checkpoint = {
                "format_version": 2,
                "kind": "mert_contrastive_two_stage_crops",
                "task": args.task,
                "labels": LABELS[args.task],
                "model_id": checkpoint["model_id"],
                "revision": checkpoint["revision"],
                "seconds": 30.0,
                "crop_seconds": checkpoint["crop_seconds"],
                "num_crops": 3,
                "projection_args": checkpoint["model_args"],
                "projection_state": checkpoint["state_dict"],
                "classifier_args": classifier_args,
                "classifier_state": cpu_state_dict(classifier),
                "stage1_checkpoint": str(checkpoint_path),
                "stage1_epoch": checkpoint["epoch"],
                "epoch": epoch,
                "seed": args.seed,
                "train_ids": [row["sample_id"] for row in train_rows],
                "validation_ids": [row["sample_id"] for row in val_rows],
                "training_loss": "stage1_era_contrastive_then_probe_cross_entropy",
                "temperature": checkpoint["temperature"],
                "metrics": score,
            }
            torch.save(final_checkpoint, output / "best.pt")
            save_json(output / "validation_metrics.json", entry)
            save_confusion(score, output / "validation_confusion.png")
        save_json(output / "history.json", history)
        if stale >= args.classifier_patience:
            break
    return global_best


def run_probe(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(f"Choose a new output directory: {output / 'best.pt'} exists")
    source = Path(args.stage1)
    if args.probe_all:
        checkpoint_paths = sorted(source.glob("epoch_*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {source}")
    elif source.is_file():
        checkpoint_paths = [source]
    else:
        raise FileNotFoundError(f"Stage1 checkpoint not found: {source}")

    first = load_stage1_checkpoint(checkpoint_paths[0], args.task)
    if first.get("kind") != "mert_contrastive_stage1_crops":
        raise ValueError("Probe requires crop stage1 checkpoints from this script")
    train_rows, val_rows = read_probe_rows(args)
    if first["train_ids"] != [row["sample_id"] for row in train_rows]:
        raise ValueError("Probe training split does not match stage1 checkpoint")
    labels = LABELS[args.task]
    device = get_device(args.device)
    all_features, resolved_revision, input_dim = extract_crop_features(
        train_rows + val_rows, first["model_id"], first["revision"],
        first["crop_seconds"], device, args.extract_batch_size)
    if input_dim != first["model_args"]["input_dim"] or resolved_revision != first["revision"]:
        raise ValueError("MERT configuration does not match stage1 checkpoint")
    train_crop_count = 3 * len(train_rows)
    x_train, x_val = all_features[:train_crop_count], all_features[train_crop_count:]
    y_train = torch.tensor(
        [labels.index(row["label"]) for row in train_rows], dtype=torch.long
    ).repeat_interleave(3)
    y_val = torch.tensor(
        [labels.index(row["label"]) for row in val_rows], dtype=torch.long
    ).repeat_interleave(3)
    save_json(output / "probe_config.json", vars(args) | {
        "labels": labels, "crop_seconds": first["crop_seconds"],
        "resolved_revision": resolved_revision,
    })

    history = []
    global_best = (-1.0, float("inf"))
    expected = (first["model_id"], first["revision"], first["crop_seconds"],
                first["model_args"], first["train_ids"])
    for checkpoint_path in checkpoint_paths:
        checkpoint = load_stage1_checkpoint(checkpoint_path, args.task)
        identity = (checkpoint["model_id"], checkpoint["revision"],
                    checkpoint["crop_seconds"], checkpoint["model_args"],
                    checkpoint["train_ids"])
        if identity != expected:
            raise ValueError(f"Incompatible stage1 checkpoint: {checkpoint_path}")
        print(f"Probing stage1 checkpoint: {checkpoint_path}", flush=True)
        global_best = probe_checkpoint(
            args, checkpoint_path, x_train, y_train, x_val, y_val,
            train_rows, val_rows, output, global_best, history, device)

    best_path = output / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Probe completed without producing best.pt")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    print(f"Global best probe: stage1_epoch={best['stage1_epoch']} "
          f"probe_epoch={best['epoch']} metrics={best['metrics']}", flush=True)

    if args.visualize:
        projection = MLPClassifier(**best["projection_args"])
        projection.load_state_dict(best["projection_state"], strict=True)
        projection.to(device).eval()
        z_all = encode_batches(
            projection, torch.cat((x_train, x_val)), device,
            args.extract_batch_size)
        paths = save_embedding_visualizations(
            {"h": torch.cat((x_train, x_val)), "z": z_all},
            torch.cat((y_train, y_val)), labels, args, output,
            prefix="contrastive_crops_probe",
            sample_ids=[f"{row['sample_id']}#crop{crop}"
                        for row in train_rows + val_rows for crop in range(3)])
        print("Saved embedding visualizations: " +
              ", ".join(str(path) for path in paths.values()))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage1", "probe"])
    parser.add_argument("--task", required=True, choices=LABELS)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--crop-seconds", type=float, default=10)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recordings-per-class", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", "--tau", type=float, default=0.1)
    parser.add_argument("--stage1")
    parser.add_argument("--probe-all", action="store_true")
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    add_visualization_args(parser)
    args = parser.parse_args()
    resolve_manifests(args, parser, require_validation=args.mode == "probe")
    if args.extract_batch_size < 1 or not 0 < args.crop_seconds <= 30:
        parser.error("extract-batch-size must be positive and crop-seconds in (0, 30]")
    if args.mode == "stage1":
        if min(args.epochs, args.batch_size, args.hidden_dim,
               args.recordings_per_class) < 1:
            parser.error("epochs, batch-size, hidden-dim and recordings-per-class must be positive")
        if args.batches_per_epoch < 0:
            parser.error("batches-per-epoch must be non-negative")
        if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
            parser.error("Invalid stage1 optimizer or dropout parameters")
        if args.temperature <= 0:
            parser.error("temperature must be positive")
        run_stage1(args)
    else:
        if not args.stage1:
            parser.error("probe mode requires --stage1")
        if min(args.classifier_hidden_dim, args.classifier_batch_size,
               args.classifier_epochs, args.classifier_patience) < 1:
            parser.error("Invalid classifier dimensions or training parameters")
        if args.classifier_lr <= 0 or args.classifier_weight_decay < 0:
            parser.error("Invalid classifier optimizer parameters")
        run_probe(args)


if __name__ == "__main__":
    main()
