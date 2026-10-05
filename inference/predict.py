"""Run inference for Task A and B with independently selected checkpoints.

The checkpoint kind determines whether the task uses LoRA or frozen-MERT
inference, so the two tasks may use different model families.
"""
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
from features import extract_features
from lora_pipeline import fixed_crops, forward_h, make_lora_encoder
from models import MERTEncoder, MLPClassifier, LinearClassifier, NonLinearClassifier
from utils import get_device


LORA_KINDS = {
    "mert_lora_probe", "mert_lora_ce",
    "mert_lora_probe_30s", "mert_lora_ce_30s",
}
CONTRASTIVE_KINDS = {
    "mert_contrastive_two_stage",
    "mert_contrastive_two_stage_crops",
}


@torch.no_grad()
def extract_lora_features(rows, encoder, seconds, batch_size, device,
                          full_recording):
    if full_recording:
        dataset = AudioDataset(rows, encoder.processor.sampling_rate, seconds=30)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: pad_sequence(batch, batch_first=True))
        waves = (batch for batch in loader)
    else:
        crops = fixed_crops(rows, encoder.processor.sampling_rate, seconds)
        loader = DataLoader(
            TensorDataset(crops), batch_size=batch_size, shuffle=False)
        waves = (batch[0] for batch in loader)

    return torch.cat([
        forward_h(encoder, batch.to(device)).float().cpu()
        for batch in waves
    ])


def classifier_from_checkpoint(checkpoint):
    state = checkpoint.get("classifier_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain classifier_state")
    if "linear.weight" in state:
        model = LinearClassifier(
            state["linear.weight"].shape[1], state["linear.weight"].shape[0])
    elif "net.0.weight" in state:
        model = NonLinearClassifier(
            state["net.0.weight"].shape[1], state["net.2.weight"].shape[0])
    else:
        raise ValueError("Cannot identify classifier architecture from classifier_state")
    model.load_state_dict(state, strict=True)
    return model


def predict_lora(checkpoint, rows, args, device):
    kind = checkpoint["kind"]
    full_recording = kind.endswith("_30s")
    seconds = 30.0 if full_recording else float(checkpoint.get("crop_seconds", 10))
    if not 0 < seconds <= 30:
        raise ValueError(f"Invalid crop/recording duration: {seconds}")

    encoder, _ = make_lora_encoder(
        checkpoint["model_id"], checkpoint["revision"], device,
        checkpoint["lora"])
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(encoder.backbone, checkpoint["adapter_state"])
    encoder.eval()
    features = extract_lora_features(
        rows, encoder, seconds, args.batch_size, device, full_recording)

    model = classifier_from_checkpoint(checkpoint).eval()
    mean, std = checkpoint.get("feature_mean"), checkpoint.get("feature_std")
    if mean is not None and std is not None:
        features = (features - mean) / std
    logits = torch.cat([
        model(batch).cpu() for batch in features.split(args.batch_size)
    ])
    if not full_recording:
        probabilities = logits.softmax(-1).reshape(len(rows), 3, -1).mean(1)
        logits = probabilities.clamp_min(1e-12).log()
    return logits, checkpoint["labels"]


def predict_mert(checkpoint, rows, args, device):
    kind = checkpoint.get("kind")
    is_two_stage = kind in CONTRASTIVE_KINDS
    is_crop_two_stage = kind == "mert_contrastive_two_stage_crops"

    if is_two_stage:
        required = {
            "projection_args", "projection_state",
            "classifier_args", "classifier_state",
        }
        if checkpoint.get("format_version") != 2 or not required.issubset(checkpoint):
            raise ValueError("Invalid two-stage contrastive checkpoint")
    elif checkpoint.get("format_version") != 1 or "state_dict" not in checkpoint:
        raise ValueError("Unsupported frozen-MERT checkpoint format")

    encoder = MERTEncoder(
        checkpoint["model_id"], checkpoint["revision"]
    ).to(device).eval()
    if is_crop_two_stage:
        crops = fixed_crops(
            rows, encoder.processor.sampling_rate,
            checkpoint["crop_seconds"])
        features = torch.cat([
            forward_h(encoder, batch.to(device)).float().cpu()
            for batch in crops.split(args.extract_batch_size)
        ])
    else:
        features = extract_features(
            rows, encoder, args.cache_dir, checkpoint["model_id"],
            checkpoint["seconds"], args.extract_batch_size)

    if is_two_stage:
        projection = MLPClassifier(**checkpoint["projection_args"])
        projection.load_state_dict(checkpoint["projection_state"], strict=True)
        classifier = NonLinearClassifier(**checkpoint["classifier_args"])
        classifier.load_state_dict(checkpoint["classifier_state"], strict=True)
        projection.to(device).eval()
        classifier.to(device).eval()
        logits = torch.cat([
            classifier(projection.encode(batch.to(device))).cpu()
            for batch in features.split(args.batch_size)
        ])
    else:
        model = MLPClassifier(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        logits = torch.cat([
            model(batch.to(device)).cpu()
            for batch in features.split(args.batch_size)
        ])

    if is_crop_two_stage:
        probabilities = logits.softmax(-1).reshape(len(rows), 3, -1).mean(1)
        logits = probabilities.clamp_min(1e-12).log()
    return logits, checkpoint["labels"]


def predict_task(args, task, checkpoint_path, manifest, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_task(checkpoint, task, checkpoint_path)
    validate_no_training_overlap(
        checkpoint,
        load_test_rows(args.data_root, task, manifest),
        checkpoint_path,
    )
    rows = load_test_rows(args.data_root, task, manifest)
    kind = checkpoint.get("kind")
    if kind in LORA_KINDS:
        logits, labels = predict_lora(checkpoint, rows, args, device)
    elif kind in CONTRASTIVE_KINDS or kind is None:
        logits, labels = predict_mert(checkpoint, rows, args, device)
    else:
        raise ValueError(f"Unsupported checkpoint kind {kind!r}: {checkpoint_path}")

    ranked = rank_predictions(logits, labels)
    return {
        row["sample_id"]: top3 for row, top3 in zip(rows, ranked)
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    add_prediction_args(parser)
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--extract-batch-size", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.extract_batch_size) < 1:
        parser.error("batch sizes must be positive")

    device = get_device(args.device)
    predictions = {}
    for task, checkpoint_path, manifest in (
        ("A", args.checkpoint_a, args.manifest_a),
        ("B", args.checkpoint_b, args.manifest_b),
    ):
        predictions[f"dataset_{task}"] = predict_task(
            args, task, checkpoint_path, manifest, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_predictions(args.output, predictions, args.template)


if __name__ == "__main__":
    main()
