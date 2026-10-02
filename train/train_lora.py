"""MERT-v2 LoRA training: SupCon stage, linear probe, and LoRA+CE baseline.

Examples:
  python train/train_lora.py stage1 --task A --output-dir runs/A_lora_s1
  python train/train_lora.py probe --task A --stage1 runs/A_lora_s1/epoch_010.pt --output-dir runs/A_lora_probe
  python train/train_lora.py ce --task A --output-dir runs/A_lora_ce
"""
from __future__ import annotations
import argparse, json, math, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv

from dataset import check_disjoint, read_manifest, TwoViewDataset, ClassBalancedBatchSampler
from lora_pipeline import (fixed_crops, forward_h, make_lora_encoder, package_versions,
    save_json, trainable_report)
from utils import LABELS, MODEL_ID, get_device, metrics, save_confusion, seed_everything, compute_svm_metrics, combine_and_record_scores
from peft import TaskType, PeftType

from models import ProjectionHead, LinearClassifier, NonLinearClassifier
from loss import supcon_loss
from visualize import add_visualization_args, save_embedding_visualizations

torch.serialization.add_safe_globals([TaskType, PeftType])

def rows_for(args):
    root = Path(args.data_root)
    manifest = args.manifest or str(root / f"dataset_{args.task}" / "manifest.csv")
    train = read_manifest(args.train_manifest or manifest, args.data_root, args.task, "train")
    val = read_manifest(args.val_manifest or manifest, args.data_root, args.task, "validation")
    check_disjoint(train, val)
    label_map = {x: i for i, x in enumerate(LABELS[args.task])}
    for r in train + val: r["label_index"] = label_map[r["label"]]
    return train, val, label_map


def device_autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.autocast("cpu", enabled=False)


def load_stage1(path, device, config_override=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["lora"])
    if config_override: cfg.update(config_override)
    enc, report = make_lora_encoder(ckpt["model_id"], ckpt["revision"], device, cfg)
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(enc.backbone, ckpt["adapter_state"])
    return ckpt, enc, report


def save_stage1(path, epoch, args, encoder, projection, report):
    from peft import get_peft_model_state_dict
    torch.save({"format_version": 2, "kind": "mert_lora_stage1", "task": args.task,
        "labels": LABELS[args.task], "model_id": args.model_id, "revision": encoder.revision,
        "adapter_epoch": epoch, "adapter_state": {k: v.cpu() for k, v in get_peft_model_state_dict(encoder.backbone).items()},
        "projection_state": {k: v.detach().cpu() for k, v in projection.state_dict().items()},
        "lora": args.lora, "crop_seconds": args.crop_seconds, "pooling": "feature_attention_mask_mean",
        "temperature": args.temperature, "seed": args.seed, "reports": report,
        "package_versions": package_versions()}, path)


@torch.no_grad()
def visualize_lora(args, encoder, rows, output, projection=None, prefix="mert_lora"):
    """Extract deterministic fixed-crop h (and optional projection z) for plotting."""
    crops = fixed_crops(rows, encoder.processor.sampling_rate, args.crop_seconds)
    h = pooled_crops(encoder, crops, get_device(args.device), args.extract_batch_size)
    targets = torch.tensor([row["label_index"] for row in rows]).repeat_interleave(3)
    representations = {"h": h}
    if projection is not None:
        projection.eval()
        representations["z"] = torch.cat([
            projection(batch.to(get_device(args.device))).float().cpu()
            for batch in h.split(args.extract_batch_size)
        ])
    crop_ids = [f"{row['sample_id']}#crop{crop}" for row in rows for crop in range(3)]
    paths = save_embedding_visualizations(
        representations, targets, LABELS[args.task], args, output,
        prefix=prefix, sample_ids=crop_ids)
    print("Saved embedding visualizations: " + ", ".join(str(path) for path in paths.values()))


def parameter_groups(named_parameters, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "norm" in name.lower() else decay).append(p)
    return [{"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0}]


def stage1(args):
    seed_everything(args.seed); device = get_device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, _ = rows_for(args)
    encoder, report = make_lora_encoder(args.model_id, args.revision, device, args.lora)
    projection = ProjectionHead(encoder.hidden_size).to(device)
    train_report = trainable_report(encoder)
    report["trainable"] = train_report
    save_json(Path(args.output_dir) / "model_inspection.json", report)
    ds = TwoViewDataset(train, encoder.processor.sampling_rate, args.crop_seconds)
    sampler = ClassBalancedBatchSampler(train, args.classes_per_batch, args.recordings_per_class, args.batches_per_epoch)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    groups = parameter_groups(encoder.named_parameters(), args.adapter_lr, args.weight_decay)
    groups += parameter_groups(projection.named_parameters(), args.projection_lr, args.weight_decay)
    opt = torch.optim.AdamW(groups)
    total = max(1, args.epochs * len(loader)); warm = max(1, round(.1 * total))
    def lr_lambda(step):
        if step < warm: return (step + 1) / warm
        return .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    save_json(out / "config.json", vars(args) | {"revision": encoder.revision, "labels": LABELS[args.task], "package_versions": package_versions()})
    for epoch in range(1, args.epochs + 1):
        encoder.train(); projection.train(); loss_sum = 0
        for view1, view2, labels in loader:
            waves = torch.cat([view1, view2], dim=0).to(device)
            labels = labels.to(device); labels2 = torch.cat([labels, labels])
            opt.zero_grad(set_to_none=True)
            with device_autocast(device):
                h = forward_h(encoder, waves)
                z = projection(h)
            loss = supcon_loss(z, labels2, args.temperature)
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projection.parameters()), 1.0)
            opt.step(); sched.step(); loss_sum += loss.item()
        print(f"epoch={epoch} supcon={loss_sum / len(loader):.5f}", flush=True)
        if epoch % 5 == 0:
            save_stage1(out / f"epoch_{epoch:03d}.pt", epoch, args, encoder, projection, report)
    final_checkpoint = out / f"epoch_{args.epochs:03d}.pt"
    if args.epochs % 5:
        save_stage1(final_checkpoint, args.epochs, args, encoder, projection, report)
    if args.visualize:
        ckpt, encoder, _ = load_stage1(final_checkpoint, device)
        projection = ProjectionHead(encoder.hidden_size).to(device)
        projection.load_state_dict(ckpt["projection_state"])
        visualize_lora(args, encoder, train + val, out, projection, prefix="lora_stage1")


@torch.no_grad()
def pooled_crops(encoder, waves, device, batch_size=16):
    encoder.eval(); result = []
    for start in range(0, len(waves), batch_size):
        result.append(forward_h(encoder, waves[start:start + batch_size].to(device)).float().cpu())
    return torch.cat(result)


def get_classifier(classifier):
    if classifier == "linear":
        return LinearClassifier
    elif classifier == "nonlinear":
        return NonLinearClassifier
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


best = (-1, float("inf"))

def probe(args, ckpt_path):
    CLF = get_classifier(args.classifier)
    seed_everything(args.seed); device = get_device(args.device)
    train, val, _ = rows_for(args); ckpt, encoder, report = load_stage1(ckpt_path, device)
    xtr = fixed_crops(train, encoder.processor.sampling_rate, args.crop_seconds)
    xva = fixed_crops(val, encoder.processor.sampling_rate, args.crop_seconds)
    htr = pooled_crops(encoder, xtr, device, args.extract_batch_size); hva = pooled_crops(encoder, xva, device, args.extract_batch_size)
    mean, std = htr.mean(0), htr.std(0, unbiased=False).clamp_min(args.std_floor)
    htr, hva = (htr - mean) / std, (hva - mean) / std
    ytr = torch.tensor([r["label_index"] for r in train]).repeat_interleave(3)
    yva = torch.tensor([r["label_index"] for r in val]).repeat_interleave(3)
    model = CLF(htr.shape[1], 6); opt = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    loader = DataLoader(TensorDataset(htr, ytr), batch_size=args.classifier_batch_size, shuffle=True)
    global best; stale = 0; out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        model.eval();
        logits = model(hva);
        probs, score = combine_and_record_scores(logits, yva, len(val));
        print(f"epoch={epoch} {score}")
        key = (score["top1"], -score["loss"])
        if key > (best[0], -best[1]):
            best, stale = (score["top1"], score["loss"]), 0
            torch.save({"format_version": 2, "kind": "mert_lora_probe", "task": args.task, "labels": LABELS[args.task],
                "model_id": ckpt["model_id"], "revision": ckpt["revision"], "adapter_state": ckpt["adapter_state"],
                "adapter_epoch": ckpt["adapter_epoch"], "model_args": {"input_dim": htr.shape[1], "num_classes": 6},
                "classifier_state": model.state_dict(), "feature_mean": mean, "feature_std": std,
                "lora": ckpt.get("lora", args.lora),
                "crop_seconds": args.crop_seconds, "pooling": "feature_attention_mask_mean", "metrics": score,
                "train_ids": [r["sample_id"] for r in train], "validation_ids": [r["sample_id"] for r in val],
                "package_versions": package_versions()}, out / "best.pt")
            save_json(out / "validation_metrics.json", score | {"epoch": epoch})
            save_confusion(metrics(probs.log(), torch.tensor([r["label_index"] for r in val]), LABELS[args.task]), out / "validation_confusion.png")
        else: stale += 1
        if stale >= args.classifier_patience: break
    if args.visualize:
        projection = ProjectionHead(encoder.hidden_size).to(device)
        projection.load_state_dict(ckpt["projection_state"])
        visualize_lora(args, encoder, train + val, out, projection, prefix="lora_probe_backbone")


def probe_all_checkpoints(args):
    ckpt_paths = list(Path(args.stage1).glob("*.pt"))
    for ckpt_path in ckpt_paths:
        probe(args, ckpt_path)
ß

def ce_baseline(args):
    CLF = get_classifier(args.classifier)
    seed_everything(args.seed); device = get_device(args.device); train, val, _ = rows_for(args)
    encoder, report = make_lora_encoder(args.model_id, args.revision, device, args.lora)
    model = CLF(encoder.hidden_size, 6).to(device)
    ds = TwoViewDataset(train, encoder.processor.sampling_rate, args.crop_seconds)
    sampler = ClassBalancedBatchSampler(train, args.classes_per_batch, args.recordings_per_class, args.batches_per_epoch)
    loader = DataLoader(ds, batch_sampler=sampler)
    groups = parameter_groups(encoder.named_parameters(), args.adapter_lr, args.weight_decay)
    groups += parameter_groups(model.named_parameters(), args.classifier_lr, args.classifier_weight_decay)
    opt = torch.optim.AdamW(groups)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); best = -1
    for epoch in range(1, args.ce_epochs + 1):
        encoder.train(); model.train()
        if epoch <= args.ce_warmup:
            for p in encoder.parameters(): p.requires_grad = False
        else:
            for n, p in encoder.named_parameters(): p.requires_grad = "lora_" in n
        for v1, v2, y in loader:
            waves = torch.cat([v1, v2]).to(device); yy = y.to(device).repeat(2)
            opt.zero_grad(set_to_none=True); h = forward_h(encoder, waves); loss = F.cross_entropy(model(h), yy); loss.backward(); torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(model.parameters()), 1); opt.step()
        # Validation uses the shared fixed-crop recording-level inference rule.
        x = fixed_crops(val, encoder.processor.sampling_rate, args.crop_seconds); h = pooled_crops(encoder, x, device, args.extract_batch_size).to(device)
        y = torch.tensor([r["label_index"] for r in val]).repeat_interleave(3); model.eval(); probs, score = combine_and_record_scores(model(h).cpu(), y, len(val)); print(f"epoch={epoch} {score}")
        if score["top1"] > best:
            best = score["top1"]
            from peft import get_peft_model_state_dict
            torch.save({"format_version": 2, "kind": "mert_lora_ce", "task": args.task, "labels": LABELS[args.task], "model_id": args.model_id, "revision": encoder.revision, "adapter_state": {k:v.cpu() for k,v in get_peft_model_state_dict(encoder.backbone).items()}, "classifier_state": {k:v.cpu() for k,v in model.state_dict().items()}, "lora": args.lora, "crop_seconds": args.crop_seconds, "pooling": "feature_attention_mask_mean", "metrics": score, "package_versions": package_versions()}, out / "best.pt")
            save_json(out / "validation_metrics.json", score | {"epoch": epoch})
            save_confusion(metrics(probs.log(), torch.tensor([r["label_index"] for r in val]), LABELS[args.task]), out / "validation_confusion.png")
    if args.visualize:
        best_ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(encoder.backbone, best_ckpt["adapter_state"])
        visualize_lora(args, encoder, train + val, out, projection=None, prefix="lora_ce")


def main():
    p = argparse.ArgumentParser();
    p.add_argument("mode", choices=["stage1", "probe", "ce"]);
    p.add_argument("--task", choices=LABELS, required=True);
    p.add_argument("--data-root", default="data/raw");
    p.add_argument("--manifest");
    p.add_argument("--train-manifest");
    p.add_argument("--val-manifest");
    p.add_argument("--output-dir", required=True);
    p.add_argument("--model-id", default=MODEL_ID);
    p.add_argument("--revision", default="main");
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"]);
    p.add_argument("--seed", type=int, default=42);
    p.add_argument("--crop-seconds", type=float, default=10);
    p.add_argument("--extract-batch-size", type=int, default=16)
    p.add_argument("--classes-per-batch", type=int, default=6);
    p.add_argument("--recordings-per-class", type=int, default=2);
    p.add_argument("--batches-per-epoch", type=int, default=0);
    p.add_argument("--temperature", type=float, default=.1);
    p.add_argument("--adapter-lr", type=float, default=1e-4);
    p.add_argument("--projection-lr", type=float, default=3e-4);
    p.add_argument("--weight-decay", type=float, default=.01);
    p.add_argument("--epochs", type=int, default=20);
    p.add_argument("--lora", type=json.loads, default={"r":8,"alpha":16,"dropout":.05})
    p.add_argument("--stage1");
    p.add_argument("--std-floor", type=float, default=1e-6);
    p.add_argument("--classifier", type=str, default="linear", choices=["linear", "nonlinear"]);
    p.add_argument("--classifier-lr", type=float, default=1e-5);
    p.add_argument("--classifier-weight-decay", type=float, default=1e-4);
    p.add_argument("--classifier-batch-size", type=int, default=64);
    p.add_argument("--classifier-epochs", type=int, default=50);
    p.add_argument("--classifier-patience", type=int, default=15);
    p.add_argument("--ce-epochs", type=int, default=50);
    p.add_argument("--ce-warmup", type=int, default=4)
    p.add_argument("--probe-all", action="store_true")
    p.add_argument("--cache-dir", default="data/features")
    add_visualization_args(p)
    args = p.parse_args(); load_dotenv()
    if args.mode == "stage1": stage1(args)
    elif args.mode == "probe":
        if not args.stage1: p.error("probe requires --stage1")

        if args.probe_all:
            probe_all_checkpoints(args, args.stage1)
        else:
            probe(args, args.stage1)
    else: ce_baseline(args)

if __name__ == "__main__": main()
