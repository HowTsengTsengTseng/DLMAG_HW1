"""Evaluate NVIDIA Audio Flamingo 3 on labeled dataset splits.

The model is prompted to rank all task labels. This gives both top-1 and
top-3 metrics from one generation while preserving a valid top-1 answer for
every sample, including generations that fail the output format.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from dotenv import load_dotenv
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import read_manifest
from utils import LABELS, save_json


MODEL_ID = "nvidia/audio-flamingo-3-hf"


def manifest_rows(data_root, task, manifest, split):
    if manifest:
        path = Path(manifest)
    else:
        candidates = [
            Path(data_root) / f"dataset_{task}" / "manifest.csv",
            Path("data/manifests") / f"{task}_{split}.csv",
        ]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Cannot find {split} manifest for task {task}; "
                f"pass --manifest-{task.lower()}")
    return read_manifest(path, data_root, task, split)


def prompt_for(task, labels, design):
    label_text = ", ".join(labels)
    if task == "A":
        task_description = "the recording's release decade"
    else:
        task_description = "the recording's release market/country"
    if design == "json":
        return (
            "Listen to the audio and estimate {description}. "
            "Choose only from these labels: {labels}.\n"
            "Return ONLY this JSON object, using double quotes and exactly "
            "three labels: {{\"top3_labels\":[\"label1\",\"label2\",\"label3\"]}}.\n"
            "Order the three labels from most likely to least likely. "
            "Do not output all six labels, explanations, markdown, or extra text."
        ).format(description=task_description, labels=label_text)
    if design == "ranked":
        return (
            "Listen to the audio and estimate {description}. "
            "Allowed labels: {labels}.\n"
            "Output exactly three labels, in descending likelihood, separated "
            "by >. Do not output all six labels, explanations, or extra text. "
            "Example: label1 > label2 > label3"
        ).format(description=task_description, labels=label_text)
    raise ValueError(f"Unknown prompt design: {design}")


def normalize_label(value, labels):
    value = str(value).strip().strip("'\"`.,:;()[]{}")
    for label in labels:
        if value.casefold() == label.casefold():
            return label
    return None


def labels_in_text(text, labels):
    found = []
    for match in re.finditer(
        r"(?<!\w)(?:" + "|".join(
            re.escape(label) for label in sorted(labels, key=len, reverse=True)
        ) + r")(?!\w)",
        text,
        flags=re.IGNORECASE,
    ):
        label = normalize_label(match.group(0), labels)
        if label and label not in found:
            found.append(label)
    return found


def parse_output(text, labels, design):
    """Return a complete ranking, validity, and a human-readable reason."""
    candidates = []
    reason = "ok"
    if design == "json":
        try:
            parsed = json.loads(text.strip())
            values = None
            if isinstance(parsed, dict):
                values = parsed.get("top3_labels", parsed.get("ranked_labels"))
            if not isinstance(values, list):
                reason = "missing-ranked_labels"
            else:
                candidates = [normalize_label(value, labels) for value in values]
                candidates = [value for value in candidates if value]
                if len(candidates) < 3 or len(set(candidates)) < 3:
                    reason = "incomplete-or-duplicate-ranking"
        except (json.JSONDecodeError, TypeError):
            # AF3 frequently emits valid Python-style dictionaries with single
            # quotes even when JSON was requested. Accept that unambiguous
            # representation, but still enforce the same label checks.
            try:
                parsed = ast.literal_eval(text.strip())
                values = parsed.get("top3_labels", parsed.get("ranked_labels")) \
                    if isinstance(parsed, dict) else None
                if not isinstance(values, list):
                    reason = "missing-ranked_labels"
                else:
                    candidates = [normalize_label(value, labels) for value in values]
                    candidates = [value for value in candidates if value]
                    if len(candidates) < 3 or len(set(candidates)) < 3:
                        reason = "incomplete-or-duplicate-ranking"
                    else:
                        reason = "ok-python-literal"
            except (SyntaxError, ValueError, TypeError):
                reason = "invalid-json"
    else:
        candidates = labels_in_text(text, labels)
        if len(candidates) < 3:
            reason = "incomplete-ranking"

    unique = []
    for label in candidates:
        if label not in unique:
            unique.append(label)
    if reason not in {"ok", "ok-python-literal"} or len(unique) < 3:
        return list(labels), False, reason or "no-label"
    return unique[:3], True, "ok" if reason == "ok" else reason


def load_model(model_id, device, dtype):
    try:
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Audio Flamingo 3 requires Transformers with native AF3 support "
            "(transformers>=5.0.0rc1). The project is using an older "
            "Transformers version; reinstall dependencies before running "
            "validate_alm.py."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id, **kwargs).eval()
    return processor, model


@torch.no_grad()
def generate_answer(processor, model, audio_path, prompt, max_new_tokens):
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": str(audio_path)},
        ],
    }]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    # AF3 is loaded in fp16/bfloat16 on CUDA, while the processor returns
    # audio features as float32 by default. Cast floating-point inputs to the
    # model dtype to keep the audio tower convolution dtype-consistent;
    # integer token/mask tensors are left unchanged by BatchFeature.to().
    inputs = inputs.to(model.device, dtype=model.dtype)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                               do_sample=False)
    continuation = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()


def evaluate_design(args, processor, model, design, task_rows):
    results = {}
    per_task = {}
    for task, rows in task_rows.items():
        labels = LABELS[task]
        prompt = prompt_for(task, labels, design)
        records = []
        invalid = 0
        confusion = [[0 for _ in labels] for _ in labels]
        top1_correct = 0
        top3_correct = 0

        for index, row in enumerate(rows, start=1):
            raw = generate_answer(
                processor, model, row["path"], prompt, args.max_new_tokens)
            ranking, valid, reason = parse_output(raw, labels, design)
            if not valid:
                invalid += 1
            target = row["label"]
            predicted = ranking[0]
            top1_correct += predicted == target
            top3_correct += target in ranking[:3]
            confusion[labels.index(target)][labels.index(predicted)] += 1
            records.append({
                "sample_id": row["sample_id"],
                "target": target,
                "prediction": predicted,
                "top3_predictions": ranking[:3],
                "valid_output": valid,
                "parse_reason": reason,
                "raw_output": raw,
            })
            print(f"[{design}] task={task} {index}/{len(rows)} "
                  f"sample={row['sample_id']} prediction={predicted}", flush=True)

        n = len(rows)
        # Generative evaluation has no calibrated class probabilities. The
        # reported loss is the uniform fallback loss: zero for a correct
        # top-1 answer and one for an incorrect answer, averaged over samples.
        # This makes the metric explicit rather than pretending generation
        # scores are comparable class probabilities.
        per_task[f"dataset_{task}"] = {
            "n_samples": n,
            "top1": top1_correct / n,
            "top3": top3_correct / n,
            "invalid_outputs": invalid,
            "invalid_rate": invalid / n,
            "fallback_label": labels[0],
            "confusion_matrix_labels": labels,
            "confusion_matrix": confusion,
            "records": records,
        }
    results[design] = per_task
    return results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest-a")
    parser.add_argument("--manifest-b")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-design", choices=["json", "ranked", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available())
                          else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rows = {
        "A": manifest_rows(args.data_root, "A", args.manifest_a, args.split),
        "B": manifest_rows(args.data_root, "B", args.manifest_b, args.split),
    }
    processor, model = load_model(args.model_id, device, dtype)
    designs = [args.prompt_design] if args.prompt_design != "both" else ["json", "ranked"]
    all_results = {
        "model_id": args.model_id,
        "split": args.split,
        "prompt_designs": designs,
        "tasks": {},
    }
    for design in designs:
        result = evaluate_design(args, processor, model, design, rows)
        all_results["tasks"][design] = result[design]
    save_json(args.output, all_results)
    print(f"Saved ALM validation report to {args.output}")


if __name__ == "__main__":
    main()
