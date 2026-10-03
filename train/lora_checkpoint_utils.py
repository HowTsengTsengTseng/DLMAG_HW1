"""Small shared helpers for standalone stage1-checkpoint workflows."""
from pathlib import Path

import torch

from dataset import check_disjoint, read_manifest
from lora_pipeline import forward_h, make_lora_encoder
from utils import LABELS


def rows_for(args):
    root = Path(args.data_root)
    manifest = args.manifest or str(root / f"dataset_{args.task}" / "manifest.csv")
    train = read_manifest(args.train_manifest or manifest, args.data_root, args.task, "train")
    val = read_manifest(args.val_manifest or manifest, args.data_root, args.task, "validation")
    check_disjoint(train, val)
    label_map = {label: index for index, label in enumerate(LABELS[args.task])}
    for row in train + val:
        row["label_index"] = label_map[row["label"]]
    return train, val


def load_stage1(path, device):
    from peft import PeftType, TaskType, set_peft_model_state_dict
    torch.serialization.add_safe_globals([TaskType, PeftType])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("kind") != "mert_lora_stage1":
        raise ValueError(f"Not a MERT LoRA stage1 checkpoint: {path}")
    encoder, _ = make_lora_encoder(ckpt["model_id"], ckpt["revision"], device, ckpt["lora"])
    set_peft_model_state_dict(encoder.backbone, ckpt["adapter_state"])
    return ckpt, encoder


@torch.no_grad()
def pooled_crops(encoder, waves, device, batch_size):
    encoder.eval()
    return torch.cat([
        forward_h(encoder, waves[start:start + batch_size].to(device)).float().cpu()
        for start in range(0, len(waves), batch_size)
    ])
