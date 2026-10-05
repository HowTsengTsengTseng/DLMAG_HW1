"""Frozen-MERT supervised contrastive training and nonlinear probing.

``stage1`` trains a projection head with Era Contrastive loss and saves one
checkpoint every five epochs. ``probe`` freezes a selected stage1 projection and
trains a nonlinear classifier. With ``--probe-all``, every stage1 checkpoint
in a directory is probed and the global best classifier is saved.
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

from dataset import check_disjoint, read_manifest
from features import extract_features
from loss import EraContrastiveLoss
from models import MERTEncoder, MLPClassifier, NonLinearClassifier
from utils import (
    LABELS, MODEL_ID, get_device, metrics, save_confusion, save_json,
    seed_everything,
)
from visualize import add_visualization_args, save_embedding_visualizations


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def resolve_manifests(args, parser, require_validation):
    if args.manifest:
        args.train_manifest = args.train_manifest or args.manifest
        if require_validation:
            args.val_manifest = args.val_manifest or args.manifest
    else:
        default_manifest = Path(args.data_root) / f"dataset_{args.task}" / "manifest.csv"
        if default_manifest.exists():
            args.train_manifest = args.train_manifest or str(default_manifest)
            if require_validation:
                args.val_manifest = args.val_manifest or str(default_manifest)

    if not args.train_manifest:
        parser.error("A training manifest is required")
    if require_validation and not args.val_manifest:
        parser.error("A validation manifest is required for probe mode")


def read_train_rows(args):
    rows = read_manifest(
        args.train_manifest, args.data_root, args.task, "train")
    labels = LABELS[args.task]
    if {row["label"] for row in rows} != set(labels):
        raise ValueError("Training split must contain all task labels")
    return rows


def read_probe_rows(args):
    train_rows = read_train_rows(args)
    val_rows = read_manifest(
        args.val_manifest, args.data_root, args.task, "validation")
    check_disjoint(train_rows, val_rows)
    return train_rows, val_rows


def extract_frozen_features(rows, model_id, revision, seconds, args, device,
                            with_augmentation=False):
    encoder = MERTEncoder(model_id, revision).to(device).eval()
    resolved_revision = encoder.revision
    input_dim = encoder.hidden_size
    features = extract_features(
        rows, encoder, args.cache_dir, model_id, seconds,
        args.extract_batch_size, with_augmentation)
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
        "kind": "mert_contrastive_stage1",
        "task": args.task,
        "labels": LABELS[args.task],
        "model_id": args.model_id,
        "revision": revision,
        "seconds": args.seconds,
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
    x_train, revision, input_dim = extract_frozen_features(
        train_rows, args.model_id, args.revision, args.seconds, args, device,
        with_augmentation=args.with_augmentation)
    y_train = torch.tensor(
        [labels.index(row["label"]) for row in train_rows], dtype=torch.long)

    projection_args = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": len(labels),
        "dropout": args.dropout,
    }
    projection = MLPClassifier(**projection_args)
    projection.fit_standardizer(x_train)
    projection.to(device)
    criterion = EraContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        projection.network[0].parameters(), lr=args.lr,
        weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed))

    save_json(output / "config.json", vars(args) | {
        "resolved_revision": revision,
        "labels": labels,
        "objective": "stage1_era_contrastive_loss",
    })
    history = []
    for epoch in range(1, args.epochs + 1):
        projection.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(projection.encode(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(y_train)
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
        z_train = encode_batches(
            projection, x_train, device, args.extract_batch_size)
        paths = save_embedding_visualizations(
            {"h": x_train, "z": z_train}, y_train, labels, args, output,
            prefix="contrastive_stage1",
            sample_ids=[row["sample_id"] for row in train_rows])
        print("Saved embedding visualizations: " +
              ", ".join(str(path) for path in paths.values()))


def load_stage1_checkpoint(path, task):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or \
            (checkpoint.get("kind") != "mert_contrastive_stage1" and \
            checkpoint.get("kind") != "mert_contrastive_stage1_crops"):
        raise ValueError(f"Expected a stage1 contrastive checkpoint: {path}")
    if checkpoint.get("task") != task:
        raise ValueError(
            f"Checkpoint task is {checkpoint.get('task')}, expected {task}: {path}")
    return checkpoint


def probe_checkpoint(args, checkpoint_path, x_train, y_train, x_val, y_val,
                     train_rows, val_rows, output, global_best, history,
                     device):
    seed_everything(args.seed)
    checkpoint = load_stage1_checkpoint(checkpoint_path, args.task)
    projection = MLPClassifier(**checkpoint["model_args"])
    projection.load_state_dict(checkpoint["state_dict"], strict=True)
    projection.to(device).eval()
    z_train = encode_batches(
        projection, x_train, device, args.extract_batch_size)
    z_val = encode_batches(
        projection, x_val, device, args.extract_batch_size)

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
            logits = classifier(z_val.to(device)).cpu()
            val_loss = F.cross_entropy(logits, y_val).item()
        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError("Non-finite probe validation loss")
        score = metrics(logits, y_val, LABELS[args.task]) | {"loss": val_loss}
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
            f"val_ce={val_loss:.5f} top1={score['top1'] * 100:.2f}% "
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
                "kind": "mert_contrastive_two_stage",
                "task": args.task,
                "labels": LABELS[args.task],
                "model_id": checkpoint["model_id"],
                "revision": checkpoint["revision"],
                "seconds": checkpoint["seconds"],
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
            raise FileNotFoundError(
                f"No epoch_*.pt stage1 checkpoints found in {source}")
    else:
        if not source.is_file():
            raise FileNotFoundError(f"Stage1 checkpoint not found: {source}")
        checkpoint_paths = [source]

    first_checkpoint = load_stage1_checkpoint(checkpoint_paths[0], args.task)
    train_rows, val_rows = read_probe_rows(args)
    train_ids = [row["sample_id"] for row in train_rows]
    if first_checkpoint["train_ids"] != train_ids:
        raise ValueError("Probe training split does not match the stage1 checkpoint")
    labels = LABELS[args.task]
    device = get_device(args.device)
    all_rows = train_rows + val_rows
    all_features, resolved_revision, input_dim = extract_frozen_features(
        all_rows, first_checkpoint["model_id"], first_checkpoint["revision"],
        first_checkpoint["seconds"], args, device)
    if input_dim != first_checkpoint["model_args"]["input_dim"]:
        raise ValueError("MERT feature dimension does not match stage1 checkpoint")
    if resolved_revision != first_checkpoint["revision"]:
        raise ValueError("Resolved MERT revision does not match stage1 checkpoint")
    x_train = all_features[:len(train_rows)]
    x_val = all_features[len(train_rows):]
    y_train = torch.tensor(
        [labels.index(row["label"]) for row in train_rows], dtype=torch.long)
    y_val = torch.tensor(
        [labels.index(row["label"]) for row in val_rows], dtype=torch.long)

    save_json(output / "probe_config.json", vars(args) | {
        "labels": labels,
        "model_id": first_checkpoint["model_id"],
        "resolved_revision": resolved_revision,
        "seconds": first_checkpoint["seconds"],
    })
    history = []
    global_best = (-1.0, float("inf"))
    for checkpoint_path in checkpoint_paths:
        checkpoint = load_stage1_checkpoint(checkpoint_path, args.task)
        identity = (
            checkpoint["model_id"], checkpoint["revision"],
            checkpoint["seconds"], checkpoint["model_args"],
            checkpoint["train_ids"])
        expected = (
            first_checkpoint["model_id"], first_checkpoint["revision"],
            first_checkpoint["seconds"], first_checkpoint["model_args"],
            first_checkpoint["train_ids"])
        if identity != expected:
            raise ValueError(
                f"Stage1 checkpoint is incompatible with the probe sweep: {checkpoint_path}")
        print(f"Probing stage1 checkpoint: {checkpoint_path}", flush=True)
        global_best = probe_checkpoint(
            args, checkpoint_path, x_train, y_train, x_val, y_val,
            train_rows, val_rows, output, global_best, history, device)

    best_path = output / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Probe completed without producing a best.pt checkpoint")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    print(
        f"Global best probe: stage1_epoch={best['stage1_epoch']} "
        f"probe_epoch={best['epoch']} metrics={best['metrics']}", flush=True)

    if args.visualize:
        projection = MLPClassifier(**best["projection_args"])
        projection.load_state_dict(best["projection_state"], strict=True)
        projection.to(device).eval()
        x_all = torch.cat((x_train, x_val))
        y_all = torch.cat((y_train, y_val))
        z_all = encode_batches(
            projection, x_all, device, args.extract_batch_size)
        paths = save_embedding_visualizations(
            {"h": x_all, "z": z_all}, y_all, labels, args, output,
            prefix="contrastive_probe",
            sample_ids=[row["sample_id"] for row in all_rows])
        print("Saved embedding visualizations: " +
              ", ".join(str(path) for path in paths.values()))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage1", "probe"])
    parser.add_argument("--task", required=True, choices=LABELS)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest", help="Combined manifest CSV")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Stage1 batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Stage1 epochs")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", "--tau", type=float, default=0.1)
    parser.add_argument("--with-augmentation", action="store_true")
    parser.add_argument("--stage1",
                        help="Stage1 checkpoint file, or directory with --probe-all")
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
    if args.extract_batch_size < 1:
        parser.error("extract-batch-size must be positive")
    if args.mode == "stage1":
        if min(args.epochs, args.batch_size, args.hidden_dim) < 1:
            parser.error("epochs, batch-size and hidden-dim must be positive")
        if args.lr <= 0 or args.weight_decay < 0:
            parser.error("Invalid stage1 optimizer parameters")
        if not 0 <= args.dropout < 1 or args.temperature <= 0:
            parser.error("dropout must be in [0, 1) and temperature must be positive")
        if not 0 < args.seconds <= 30:
            parser.error("seconds must be in (0, 30]")
        run_stage1(args)
    else:
        if not args.stage1:
            parser.error("probe mode requires --stage1")
        if min(args.classifier_hidden_dim, args.classifier_batch_size,
               args.classifier_epochs, args.classifier_patience) < 1:
            parser.error("classifier dimensions, batch size, epochs and patience must be positive")
        if args.classifier_lr <= 0 or args.classifier_weight_decay < 0:
            parser.error("Invalid classifier optimizer parameters")
        run_probe(args)


if __name__ == "__main__":
    main()
