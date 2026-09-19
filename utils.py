"""Shared configuration, reproducibility, and classification metrics."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

LABELS = {
    "A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
MODEL_ID = "m-a-p/MERT-v2-30s"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name="auto"):
    # CPU fallback is deliberate: MERT's audio frontend is best supported on CUDA/CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(logits, targets, labels):
    ranked = logits.argsort(dim=1, descending=True)
    return {
        "n_samples": len(targets),
        "top1": (ranked[:, 0] == targets).float().mean().item(),
        "top3": (ranked[:, :3] == targets[:, None]).any(dim=1).float().mean().item(),
        "labels": labels,
        "confusion_matrix_type": "counts; rows=true, columns=predicted",
        "confusion_matrix": confusion_matrix(targets.cpu(), ranked[:, 0].cpu(), labels=list(range(len(labels)))).tolist(),
    }


def save_confusion(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=result["labels"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation confusion matrix (counts)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
