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

from dataset import check_disjoint, read_manifest
from lora_pipeline import (ClassBalancedBatchSampler, LinearClassifier, ProjectionHead,
    TwoViewDataset, fixed_crops, forward_h, make_lora_encoder, package_versions,
    save_json, supcon_loss, trainable_report)
from utils import LABELS, MODEL_ID, get_device, metrics, save_confusion, seed_everything, compute_svm_metrics
from features import extract_features
from peft import TaskType, PeftType

from sklearn.svm import LinearSVC
import numpy as np
from models import SVMClassifier

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


@torch.no_grad()
def pooled_crops(encoder, waves, device, batch_size=16):
    encoder.eval(); result = []
    for start in range(0, len(waves), batch_size):
        result.append(forward_h(encoder, waves[start:start + batch_size].to(device)).float().cpu())
    return torch.cat(result)


def recording_scores(logits, labels, n_records):
    probs = logits.softmax(-1).reshape(n_records, 3, -1).mean(1)
    y = labels[::3][:n_records]
    top1 = (probs.argmax(1) == y).float().mean().item()
    top3 = (probs.topk(3, 1).indices == y[:, None]).any(1).float().mean().item()
    ce = (-probs[torch.arange(n_records), y].clamp_min(1e-8).log()).mean().item()
    return probs, {"top1": top1, "top3": top3, "loss": ce}

best = (-1, float("inf"))

def probe(args, ckpt_path):
    seed_everything(args.seed); device = get_device(args.device)
    train, val, _ = rows_for(args); ckpt, encoder, report = load_stage1(ckpt_path, device)
    xtr = fixed_crops(train, encoder.processor.sampling_rate, args.crop_seconds)
    xva = fixed_crops(val, encoder.processor.sampling_rate, args.crop_seconds)
    htr = pooled_crops(encoder, xtr, device, args.extract_batch_size); hva = pooled_crops(encoder, xva, device, args.extract_batch_size)
    mean, std = htr.mean(0), htr.std(0, unbiased=False).clamp_min(args.std_floor)
    htr, hva = (htr - mean) / std, (hva - mean) / std
    ytr = torch.tensor([r["label_index"] for r in train]).repeat_interleave(3)
    yva = torch.tensor([r["label_index"] for r in val]).repeat_interleave(3)
    model = LinearClassifier(htr.shape[1], 6); opt = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    loader = DataLoader(TensorDataset(htr, ytr), batch_size=args.classifier_batch_size, shuffle=True)
    global best; stale = 0; out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        model.eval();
        logits = model(hva);
        probs, score = recording_scores(logits, yva, len(val));
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


def probe_all_checkpoints(args):
    ckpt_paths = list(Path(args.stage1).glob("*.pt"))
    for ckpt_path in ckpt_paths:
        probe(args, ckpt_path)


def probe_with_svm(args):
    seed_everything(args.seed); device = get_device(args.device)
    train_rows, val_rows, labels = rows_for(args);
    ckpt, encoder, report = load_stage1(args.stage1, device)
    revision, input_dim = encoder.revision, encoder.hidden_size
    features = [extract_features(rows, encoder, args.cache_dir, args.model_id) for rows in (train_rows, val_rows)]
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    labels = LABELS[args.task]
    x_train, x_val = features
    y_train = np.array([labels.index(r["label"]) for r in train_rows])
    y_val = np.array([labels.index(r["label"]) for r in val_rows])

    c_candidates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    best_c = None
    best_score = -1.0
    best_top3 = -1.0
    best_clf = None
    best_metrics = None
    grid_results = []

    mean = x_train.mean(0); std = x_train.std(0, unbiased=False).clamp_min(10e-6)
    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - x_val.mean(0)) / x_val.std(0, unbiased=False).clamp_min(10e-6)
    for c_val in c_candidates:
        clf = LinearSVC(C=c_val, random_state=args.seed, max_iter=5000, dual="auto")
        clf.fit(x_train_norm, y_train)

        # Compute decision function scores on validation set
        val_scores = clf.decision_function(x_val_norm)
        scores = compute_svm_metrics(val_scores, y_val, labels)

        grid_results.append({"C": c_val, "top1": scores["top1"], "top3": scores["top3"]})
        print(f"C={c_val:>7.3f} | Val Top-1: {scores['top1']*100:.2f}% | Val Top-3: {scores['top3']*100:.2f}%")

        # Select by Top-1 accuracy (tie-break on Top-3)
        if (scores["top1"] > best_score) or (scores["top1"] == best_score and scores["top3"] > best_top3):
            best_score = scores["top1"]
            best_top3 = scores["top3"]
            best_c = c_val
            best_clf = clf
            best_metrics = scores

    print(f"\nBest C={best_c} -> Val Top-1: {best_score*100:.2f}%, Val Top-3: {best_top3*100:.2f}%")

    # Save PyTorch-compatible SVMClassifier for seamless test.py evaluation
    model_args = dict(input_dim=input_dim, num_classes=6)
    py_model = SVMClassifier(**model_args)
    py_model.feature_mean.copy_(mean)
    py_model.feature_std.copy_(std)

    py_model.set_weights(best_clf.coef_, best_clf.intercept_)

    # Verify exact equivalence between PyTorch model and sklearn LinearSVC
    py_model.eval()
    with torch.no_grad():
        py_logits = py_model(x_val).numpy()
    sk_logits = best_clf.decision_function(x_val_norm)
    assert np.allclose(py_logits, sk_logits, atol=1e-4), "PyTorch and sklearn decision functions diverged!"

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_class": "SVMClassifier",
        "task": args.task,
        "labels": labels,
        "model_id": args.model_id,
        "revision": revision,
        # "seconds": args.seconds,
        "model_args": model_args,
        "state_dict": {k: v.detach().cpu() for k, v in py_model.state_dict().items()},
        "best_C": best_c,
        # "seed": args.seed,
        "train_ids": [r["sample_id"] for r in x_train],
        "validation_ids": [r["sample_id"] for r in x_val],
    }

    torch.save(checkpoint, out / "best.pt")
    save_json(out / "config.json", vars(args) | {
        "resolved_revision": revision,
        "labels": labels,
        "best_C": best_c,
        "grid_results": grid_results,
    })
    save_json(out / "validation_metrics.json", best_metrics | {"best_C": best_c})
    save_confusion(best_metrics, out / "validation_confusion.png")

    # Also save raw sklearn model via joblib if available
    try:
        import joblib
        joblib.dump(best_clf, out / "best_svm.joblib")
    except Exception:
        pass

    print(f"Saved PyTorch-compatible checkpoint: {out / 'best.pt'}")


def ce_baseline(args):
    seed_everything(args.seed); device = get_device(args.device); train, val, _ = rows_for(args)
    encoder, report = make_lora_encoder(args.model_id, args.revision, device, args.lora)
    model = LinearClassifier(encoder.hidden_size, 6).to(device)
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
        y = torch.tensor([r["label_index"] for r in val]).repeat_interleave(3); model.eval(); probs, score = recording_scores(model(h).cpu(), y, len(val)); print(f"epoch={epoch} {score}")
        if score["top1"] > best:
            best = score["top1"]
            from peft import get_peft_model_state_dict
            torch.save({"format_version": 2, "kind": "mert_lora_ce", "task": args.task, "labels": LABELS[args.task], "model_id": args.model_id, "revision": encoder.revision, "adapter_state": {k:v.cpu() for k,v in get_peft_model_state_dict(encoder.backbone).items()}, "classifier_state": {k:v.cpu() for k,v in model.state_dict().items()}, "lora": args.lora, "crop_seconds": args.crop_seconds, "pooling": "feature_attention_mask_mean", "metrics": score, "package_versions": package_versions()}, out / "best.pt")
            save_json(out / "validation_metrics.json", score | {"epoch": epoch})
            save_confusion(metrics(probs.log(), torch.tensor([r["label_index"] for r in val]), LABELS[args.task]), out / "validation_confusion.png")


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
    p.add_argument("--classifier-lr", type=float, default=1e-5);
    p.add_argument("--classifier-weight-decay", type=float, default=1e-4);
    p.add_argument("--classifier-batch-size", type=int, default=64);
    p.add_argument("--classifier-epochs", type=int, default=50);
    p.add_argument("--classifier-patience", type=int, default=15);
    p.add_argument("--ce-epochs", type=int, default=50);
    p.add_argument("--ce-warmup", type=int, default=4)
    p.add_argument("--svm", action="store_true")
    p.add_argument("--probe-all", action="store_true")
    p.add_argument("--cache-dir", default="data/features")
    args = p.parse_args(); load_dotenv()
    if args.mode == "stage1": stage1(args)
    elif args.mode == "probe":
        if not args.stage1: p.error("probe requires --stage1")
        if args.svm:
            probe_with_svm(args)
        elif args.probe_all:
            probe_all_checkpoints(args)
        else:
            probe(args, args.stage1)
    else: ce_baseline(args)

if __name__ == "__main__": main()
