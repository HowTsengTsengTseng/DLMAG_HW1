"""Shared test-manifest and submission-format helpers for inference scripts."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import read_manifest
from utils import LABELS, save_json


def add_prediction_args(parser):
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest-a", help="Test manifest for task A")
    parser.add_argument("--manifest-b", help="Test manifest for task B")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output", required=True,
                        help="Output JSON in the dataset_A/dataset_B top-3 submission format")
    parser.add_argument("--template", default="data/raw/prediction_format_example_NOT_ANSWERS.json",
                        help="Optional example JSON used only to validate output sample IDs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)


def load_test_rows(data_root, task, manifest=None):
    if manifest is None:
        candidates = [
            Path(data_root) / f"dataset_{task}" / "manifest.csv",
            Path("data/manifests") / f"{task}_test.csv",
        ]
        manifest_path = next((path for path in candidates if path.exists()), None)
        if manifest_path is None:
            raise FileNotFoundError(
                f"Cannot find a test manifest for task {task}; pass --manifest-{task.lower()}")
    else:
        manifest_path = Path(manifest)
    return read_manifest(manifest_path, data_root, task, "test")


def validate_checkpoint_task(checkpoint, task, path):
    if checkpoint.get("task") != task or checkpoint.get("labels") != LABELS[task]:
        raise ValueError(f"Checkpoint task/labels do not match task {task}: {path}")


def validate_no_training_overlap(checkpoint, rows, path):
    fitted_ids = set(checkpoint.get("train_ids", [])) | set(checkpoint.get("validation_ids", []))
    overlap = fitted_ids & {row["sample_id"] for row in rows}
    if overlap:
        raise ValueError(f"Test IDs overlap checkpoint training/validation IDs ({len(overlap)}): {path}")


def rank_predictions(logits, labels):
    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError(f"Expected [N, {len(labels)}] logits, got {tuple(logits.shape)}")
    ranks = logits.argsort(dim=1, descending=True)[:, :3].cpu().tolist()
    return [[labels[class_id] for class_id in row] for row in ranks]


def write_predictions(output_path, predictions, template_path=None):
    if set(predictions) != {"dataset_A", "dataset_B"}:
        raise ValueError("Submission must contain exactly dataset_A and dataset_B")
    for task_name, mapping in predictions.items():
        if any(len(top3) != 3 or len(set(top3)) != 3 for top3 in mapping.values()):
            raise ValueError(f"Every {task_name} sample needs three distinct ranked labels")

    if template_path:
        import json
        with Path(template_path).open(encoding="utf-8") as stream:
            template = json.load(stream)
        if set(template) != set(predictions):
            raise ValueError("Template must contain dataset_A and dataset_B")
        for task_name in predictions:
            if set(predictions[task_name]) != set(template[task_name]):
                raise ValueError(f"Output sample IDs do not match template: {task_name}")

    save_json(output_path, predictions)
    print(f"Saved predictions to {output_path}: " +
          ", ".join(f"{task}={len(rows)}" for task, rows in predictions.items()))
