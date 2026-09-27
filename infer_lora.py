"""Inference for LoRA stage-2 and LoRA+CE checkpoints.

The output format is the assignment's dataset_A/dataset_B top-three JSON.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from dataset import read_manifest
from lora_pipeline import LinearClassifier, fixed_crops, forward_h, make_lora_encoder
from utils import LABELS, MODEL_ID, get_device, save_json

def load_model(path, task, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("task") != task or ckpt.get("labels") != LABELS[task]: raise ValueError("checkpoint task/label order mismatch")
    enc, _ = make_lora_encoder(ckpt["model_id"], ckpt["revision"], device, ckpt.get("lora", {"r":8,"alpha":16,"dropout":.05}))
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(enc.backbone, ckpt["adapter_state"]); enc.eval()
    model = LinearClassifier(ckpt.get("model_args", {"input_dim": enc.hidden_size})["input_dim"], 6)
    model.load_state_dict(ckpt["classifier_state"]); model.to(device).eval()
    return ckpt, enc, model

@torch.no_grad()
def predict(rows, ckpt, enc, model, device):
    waves = fixed_crops(rows, enc.processor.sampling_rate, ckpt.get("crop_seconds", 10))
    values = []
    for i in range(0, len(waves), 16):
        h = forward_h(enc, waves[i:i+16].to(device)).float()
        if "feature_mean" in ckpt: h = (h - ckpt["feature_mean"].to(device)) / ckpt["feature_std"].to(device)
        values.append(model(h).softmax(-1).cpu())
    probs = torch.cat(values).reshape(len(rows), 3, -1).mean(1)
    labels = ckpt["labels"]
    return {r["sample_id"]: [labels[i] for i in probs[j].argsort(descending=True)[:3].tolist()] for j, r in enumerate(rows)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root", default="data/raw"); p.add_argument("--manifest-a"); p.add_argument("--manifest-b"); p.add_argument("--checkpoint-a", required=True); p.add_argument("--checkpoint-b", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"]); args=p.parse_args()
    device=get_device(args.device); result={}
    for task, manifest, checkpoint in (("A",args.manifest_a,args.checkpoint_a),("B",args.manifest_b,args.checkpoint_b)):
        if not manifest: manifest=f"{args.data_root}/dataset_{task}/manifest.csv"
        rows=read_manifest(manifest,args.data_root,task,"test"); ckpt,enc,model=load_model(checkpoint,task,device)
        result[f"dataset_{task}"]=predict(rows,ckpt,enc,model,device)
    save_json(args.output,result); print(f"Saved {args.output}")

if __name__ == "__main__": main()
