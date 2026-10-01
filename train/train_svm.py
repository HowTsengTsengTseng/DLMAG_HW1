"""Train an SVM classifier on frozen MERT features with automatic C search."""
import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path when running from subfolder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
import torch
from torch.nn import functional as F

from dataset import check_disjoint, read_manifest
from features import extract_features
from models import MERTEncoder, SVMClassifier
from utils import LABELS, MODEL_ID, get_device, save_confusion, save_json, seed_everything, compute_svm_metrics


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=LABELS)
    parser.add_argument("--data-root", default="data/raw", help="Root directory containing datasets (default: data/raw)")
    parser.add_argument("--manifest", help="Combined manifest CSV (e.g. data/raw/dataset_A/manifest.csv); auto-filters splits")
    parser.add_argument("--train-manifest", help="Train manifest CSV (defaults to --manifest)")
    parser.add_argument("--val-manifest", help="Validation manifest CSV (defaults to --manifest)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="Hugging Face revision; resolved commit is saved")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf"], help="SVM kernel (default: linear)")
    parser.add_argument("--c", type=float, default=None, help="SVM regularization parameter C. If omitted, searches over a grid.")
    args = parser.parse_args()

    if args.manifest:
        if not args.train_manifest:
            args.train_manifest = args.manifest
        if not args.val_manifest:
            args.val_manifest = args.manifest
    elif not (args.train_manifest and args.val_manifest):
        default_manifest = Path(args.data_root) / f"dataset_{args.task}" / "manifest.csv"
        if default_manifest.exists():
            args.train_manifest = str(default_manifest)
            args.val_manifest = str(default_manifest)
        else:
            parser.error("Must provide --manifest, both --train-manifest and --val-manifest, or have data/raw/dataset_{task}/manifest.csv")

    seed_everything(args.seed)
    device = get_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if (output / "best.pt").exists():
        raise FileExistsError(f"Choose a new output directory: {output / 'best.pt'} exists")

    train_rows = read_manifest(args.train_manifest, args.data_root, args.task, "train")
    val_rows = read_manifest(args.val_manifest, args.data_root, args.task, "validation")
    check_disjoint(train_rows, val_rows)
    labels = LABELS[args.task]
    num_classes = len(labels)
    if {r["label"] for r in train_rows} != set(labels):
        raise ValueError("Training split must contain all six task labels")

    # Extract/load MERT features
    encoder = MERTEncoder(args.model_id, args.revision).to(device)
    revision, input_dim = encoder.revision, encoder.hidden_size
    features = [extract_features(rows, encoder, args.cache_dir, args.model_id,
                                 args.seconds, args.extract_batch_size) for rows in (train_rows, val_rows)]
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    x_train, x_val = features
    y_train = np.array([labels.index(r["label"]) for r in train_rows])
    y_val = np.array([labels.index(r["label"]) for r in val_rows])

    # Feature preprocessing: standardizer (mean/std) + L2 normalization
    mean = x_train.mean(0)
    std = x_train.std(0, unbiased=False).clamp_min(1e-6)

    def preprocess(x):
        normed = (x - mean) / std
        return F.normalize(normed, p=2, dim=-1).numpy()

    x_train_norm = preprocess(x_train)
    x_val_norm = preprocess(x_val)

    # Candidate values of C to search
    c_candidates = [args.c] if args.c is not None else [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    best_c = None
    best_score = -1.0
    best_top3 = -1.0
    best_clf = None
    best_metrics = None
    grid_results = []

    print(f"Training MERT + SVM ({args.kernel} kernel)...")
    for c_val in c_candidates:
        if args.kernel == "linear":
            clf = LinearSVC(C=c_val, random_state=args.seed, max_iter=5000, dual="auto")
        else:
            clf = SVC(C=c_val, kernel="rbf", random_state=args.seed, probability=True)

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
    model_args = dict(input_dim=input_dim, num_classes=num_classes)
    py_model = SVMClassifier(**model_args)
    py_model.feature_mean.copy_(mean)
    py_model.feature_std.copy_(std)

    if args.kernel == "linear":
        py_model.set_weights(best_clf.coef_, best_clf.intercept_)

        # Verify exact equivalence between PyTorch model and sklearn LinearSVC
        py_model.eval()
        with torch.no_grad():
            py_logits = py_model(x_val).numpy()
        sk_logits = best_clf.decision_function(x_val_norm)
        assert np.allclose(py_logits, sk_logits, atol=1e-4), "PyTorch and sklearn decision functions diverged!"

    checkpoint = {
        "format_version": 1,
        "model_class": "SVMClassifier",
        "task": args.task,
        "labels": labels,
        "model_id": args.model_id,
        "revision": revision,
        "seconds": args.seconds,
        "model_args": model_args,
        "state_dict": {k: v.detach().cpu() for k, v in py_model.state_dict().items()},
        "kernel": args.kernel,
        "best_C": best_c,
        "seed": args.seed,
        "train_ids": [r["sample_id"] for r in train_rows],
        "validation_ids": [r["sample_id"] for r in val_rows],
    }

    torch.save(checkpoint, output / "best.pt")
    save_json(output / "config.json", vars(args) | {
        "resolved_revision": revision,
        "labels": labels,
        "best_C": best_c,
        "grid_results": grid_results,
    })
    save_json(output / "validation_metrics.json", best_metrics | {"best_C": best_c})
    save_confusion(best_metrics, output / "validation_confusion.png")

    # Also save raw sklearn model via joblib if available
    try:
        import joblib
        joblib.dump(best_clf, output / "best_svm.joblib")
    except Exception:
        pass

    print(f"Saved PyTorch-compatible checkpoint: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
