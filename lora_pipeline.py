"""Shared MERT-v2 LoRA, crop, pooling, SupCon, and checkpoint utilities.

This module deliberately keeps the PEFT wrapper generic: MERT's remote model
code is inspected at runtime and only verified attention projection paths are
adapted.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dataset import AudioDataset
from models import MERTEncoder


def package_versions():
    import importlib.metadata as md
    result = {}
    for name in ("torch", "transformers", "peft", "huggingface-hub", "torchaudio"):
        try:
            result[name] = md.version(name)
        except md.PackageNotFoundError:
            result[name] = "missing"
    return result


def inspect_backbone(backbone: nn.Module) -> list[str]:
    """Find exact final-four attention projection paths, failing loudly."""
    leaves = [(n, m) for n, m in backbone.named_modules() if n and not list(m.children())]
    separate = []
    fused = []
    for name, module in leaves:
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            continue
        low = name.lower()
        if any(x in low for x in ("q_proj", "query", "query_proj")):
            separate.append((name, "q"))
        if any(x in low for x in ("v_proj", "value", "value_proj")):
            separate.append((name, "v"))
        if any(x in low for x in ("qkv", "q_k_v", "query_key_value", "in_proj")):
            fused.append(name)

    # Block numbers are extracted from the path, and the last four observed
    # blocks are selected. This is validated again against the actual modules.
    def block_id(name):
        parts = name.split(".")
        ids = [int(p) for p in parts if p.isdigit()]
        return ids[-1] if ids else None

    ids = sorted({block_id(n) for n, _ in separate if block_id(n) is not None})
    if separate and len(ids) >= 4:
        wanted = set(ids[-4:])
        targets = [n for n, _ in separate if block_id(n) in wanted]
        if len(targets) >= 8:
            return sorted(set(targets))
    fused_ids = sorted({block_id(n) for n in fused if block_id(n) is not None})
    if fused and len(fused_ids) >= 4:
        wanted = set(fused_ids[-4:])
        targets = [n for n in fused if block_id(n) in wanted]
        if targets:
            return sorted(set(targets))
    raise RuntimeError(
        "Could not verify Q/V or fused QKV projections in the final four "
        f"transformer blocks. Leaf modules: {[n for n, _ in leaves][:80]}"
    )


def make_lora_encoder(model_id: str, revision: str, device: torch.device, config: dict):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT is required; install the project requirements first") from exc
    base = MERTEncoder(model_id, revision)
    targets = inspect_backbone(base.backbone)
    target_types = {type(dict(base.backbone.named_modules())[x]).__name__ for x in targets}
    lora_cfg = LoraConfig(
        r=int(config.get("r", 8)), lora_alpha=int(config.get("alpha", 16)),
        lora_dropout=float(config.get("dropout", .05)), bias="none",
        target_modules=targets, task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_backbone = get_peft_model(base.backbone, lora_cfg)
    # PEFT's feature-extraction wrapper is text-oriented and forwards an
    # ``input_ids`` keyword. MERT-v2's verified audio signature is
    # ``(input_values, attention_mask, output_hidden_states, return_dict)``.
    # Keep PEFT's LoraModel and state handling, but route the audio arguments
    # directly through it.
    from peft import PeftModelForFeatureExtraction
    if isinstance(peft_backbone, PeftModelForFeatureExtraction):
        class AudioPeftModel(PeftModelForFeatureExtraction):
            def forward(self, input_values=None, attention_mask=None, **kwargs):
                kwargs.pop("input_ids", None)
                return self.base_model(input_values=input_values, attention_mask=attention_mask, **kwargs)
        peft_backbone.__class__ = AudioPeftModel
    # PEFT freezes the original parameters; enforce this because custom model
    # code may expose parameters outside the wrapped transformer.
    for name, p in peft_backbone.named_parameters():
        p.requires_grad = ("lora_" in name)
    base.backbone = peft_backbone
    base.to(device)
    report = {"targets": targets, "target_module_types": sorted(target_types),
              "base_model_id": model_id, "revision": base.revision,
              "lora": lora_cfg.to_dict()}
    return base, report


def trainable_report(model: nn.Module):
    rows = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(n for _, n in rows),
            "names": [n for n, _ in rows]}


def masked_pool(output):
    hidden = output.last_hidden_state
    mask = getattr(output, "feature_attention_mask", None)
    if mask is None:
        raise RuntimeError("MERT output has no feature_attention_mask; refusing waveform-mask pooling")
    mask = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


def forward_h(encoder, waveforms):
    inputs = encoder.processor(waveforms, sampling_rate=encoder.processor.sampling_rate,
                               padding=True, return_attention_mask=True, return_tensors="pt")
    # MERT-v2's processor can retain a singleton channel dimension for a list
    # of waveform arrays; its verified custom forward requires [batch, samples].
    if inputs["input_values"].ndim == 3 and inputs["input_values"].shape[0] == 1:
        inputs["input_values"] = inputs["input_values"].squeeze(0)
    if inputs["attention_mask"].ndim == 2 and inputs["attention_mask"].shape[0] == 1:
        inputs["attention_mask"] = inputs["attention_mask"].t().expand(-1, inputs["input_values"].shape[1])
    inputs = inputs.to(next(encoder.backbone.parameters()).device)
    return masked_pool(encoder.backbone(**inputs, return_dict=True))


def fixed_crops(rows, sample_rate=24000, crop_seconds=10):
    base = AudioDataset(rows, sample_rate, seconds=30)
    crop = round(sample_rate * crop_seconds)
    result = []
    for i in range(len(rows)):
        wave = torch.from_numpy(base[i])
        for start in (0, crop, 2 * crop):
            part = wave[start:start + crop]
            result.append(F.pad(part, (0, max(0, crop - len(part)))))
    return torch.stack(result)


def cache_key(checkpoint, crop_policy):
    blob = json.dumps({"revision": checkpoint["revision"], "adapter": checkpoint["adapter_epoch"],
                       "crop": crop_policy, "pooling": "feature_attention_mask_mean"}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
