"""Predict test top-3 labels from LoRA probe or LoRA+CE checkpoints."""
import argparse

from dotenv import load_dotenv
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from common import (
    add_prediction_args,
    load_test_rows,
    rank_predictions,
    validate_checkpoint_task,
    validate_no_training_overlap,
    write_predictions,
)
from dataset import AudioDataset
from lora_pipeline import fixed_crops, forward_h, make_lora_encoder
from models import LinearClassifier, NonLinearClassifier
from utils import get_device


@torch.no_grad()
def extract_h(rows, encoder, seconds, batch_size, device, full_recording):
    if full_recording:
        dataset = AudioDataset(rows, encoder.processor.sampling_rate, seconds=30)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: pad_sequence(batch, batch_first=True))
        waves = (batch for batch in loader)
    else:
        crops = fixed_crops(rows, encoder.processor.sampling_rate, seconds)
        loader = DataLoader(TensorDataset(crops), batch_size=batch_size, shuffle=False)
        waves = (batch[0] for batch in loader)

    vectors = [forward_h(encoder, batch.to(device)).float().cpu() for batch in waves]
    return torch.cat(vectors)


def classifier_from_checkpoint(checkpoint):
    state = checkpoint.get("classifier_state")
    if not isinstance(state, dict):
        raise ValueError("This is a LoRA stage1 representation checkpoint; run probe or CE training first")
    if "linear.weight" in state:
        model = LinearClassifier(state["linear.weight"].shape[1], state["linear.weight"].shape[0])
    elif "net.0.weight" in state:
        model = NonLinearClassifier(state["net.0.weight"].shape[1], state["net.2.weight"].shape[0])
    else:
        raise ValueError("Cannot identify the classifier architecture from classifier_state")
    model.load_state_dict(state, strict=True)
    return model


def predict_task(args, task, manifest, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_task(checkpoint, task, checkpoint_path)
    kind = checkpoint.get("kind")
    supported = {"mert_lora_probe", "mert_lora_ce", "mert_lora_probe_30s"}
    if kind not in supported:
        raise ValueError(f"Unsupported LoRA prediction checkpoint kind {kind!r}: {checkpoint_path}")
    rows = load_test_rows(args.data_root, task, manifest)
    validate_no_training_overlap(checkpoint, rows, checkpoint_path)

    full_recording = kind == "mert_lora_probe_30s"
    seconds = 30.0 if full_recording else float(checkpoint.get("crop_seconds", 10))
    if not 0 < seconds <= 30:
        raise ValueError(f"Invalid crop/recording duration in checkpoint: {seconds}")

    encoder, _ = make_lora_encoder(
        checkpoint["model_id"], checkpoint["revision"], device, checkpoint["lora"])
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(encoder.backbone, checkpoint["adapter_state"])
    encoder.eval()
    h = extract_h(rows, encoder, seconds, args.batch_size, device, full_recording)

    model = classifier_from_checkpoint(checkpoint).eval()
    mean, std = checkpoint.get("feature_mean"), checkpoint.get("feature_std")
    if mean is not None and std is not None:
        h = (h - mean) / std
    with torch.no_grad():
        logits = torch.cat([
            model(batch).cpu()
            for batch in h.split(args.batch_size)
        ])
    if not full_recording:
        probabilities = logits.softmax(-1).reshape(len(rows), 3, -1).mean(1)
        logits = probabilities.clamp_min(1e-12).log()
    ranked = rank_predictions(logits, checkpoint["labels"])
    return {
        row["sample_id"]: top3 for row, top3 in zip(rows, ranked)
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    add_prediction_args(parser)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    device = get_device(args.device)
    predictions = {}
    for task, manifest, checkpoint_path in (
        ("A", args.manifest_a, args.checkpoint_a),
        ("B", args.manifest_b, args.checkpoint_b),
    ):
        predictions[f"dataset_{task}"] = predict_task(
            args, task, manifest, checkpoint_path, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_predictions(args.output, predictions, args.template)


if __name__ == "__main__":
    main()
