"""Reusable t-SNE and UMAP plotting for precomputed representations."""
from pathlib import Path
import csv

import torch


def add_visualization_args(parser):
    """Add the shared, opt-in visualization CLI options to a training parser."""
    parser.add_argument("--visualize", action="store_true",
                        help="Save t-SNE and UMAP plots for the selected checkpoint")
    parser.add_argument("--visualize-max-samples", "--tsne-max-samples", dest="visualize_max_samples",
                        type=int, default=2000,
                        help="Maximum stratified examples per visualization")
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)


def stratified_indices(labels, max_samples, seed):
    """Select a deterministic, approximately class-balanced subset."""
    if len(labels) <= max_samples:
        return torch.arange(len(labels))
    generator = torch.Generator().manual_seed(seed)
    selected, remaining = [], []
    classes = torch.unique(labels, sorted=True)
    quota = max_samples // len(classes)
    for class_id in classes:
        indices = torch.where(labels == class_id)[0]
        order = torch.randperm(len(indices), generator=generator)
        selected.append(indices[order[:quota]])
        remaining.append(indices[order[quota:]])
    selected = torch.cat(selected)
    leftover = max_samples - len(selected)
    if leftover:
        rest = torch.cat(remaining)
        order = torch.randperm(len(rest), generator=generator)
        selected = torch.cat((selected, rest[order[:leftover]]))
    return selected.sort().values


def save_embedding_visualizations(embeddings, labels_tensor, labels, args, output_dir,
                                  prefix="embeddings", sample_ids=None):
    """Plot named 2D representations (e.g. h and z) using identical selected rows.

    ``embeddings`` maps representation names to [N, D] tensors. Every tensor and
    ``labels_tensor`` must use the same row order.
    """
    from sklearn.manifold import TSNE
    from umap import UMAP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not embeddings:
        raise ValueError("At least one embedding representation is required")
    count = len(labels_tensor)
    if any(value.ndim != 2 or len(value) != count for value in embeddings.values()):
        raise ValueError("All representations must be [N, D] tensors aligned with labels")
    if sample_ids is not None and len(sample_ids) != count:
        raise ValueError("sample_ids must be aligned with the representations and labels")
    if count < 3:
        raise ValueError("Need at least three examples to compute t-SNE and UMAP")
    max_samples = args.visualize_max_samples
    if max_samples < 3 or args.tsne_perplexity <= 0 or args.umap_n_neighbors < 2:
        raise ValueError("Visualization needs max_samples >= 3, positive perplexity, and neighbors >= 2")
    if not 0 <= args.umap_min_dist <= 1:
        raise ValueError("UMAP min_dist must be between 0 and 1")

    indices = stratified_indices(labels_tensor.cpu(), max_samples, args.seed)
    targets = labels_tensor[indices].cpu().numpy()
    selected_ids = ([str(sample_ids[i]) for i in indices.tolist()]
                    if sample_ids is not None else [str(i) for i in indices.tolist()])
    output_dir = Path(output_dir) / "visualizations" / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, tensor in embeddings.items():
        values = tensor[indices].detach().float().cpu().numpy()
        if not torch.isfinite(torch.from_numpy(values)).all():
            raise ValueError(f"Non-finite values in {name} representation")
        perplexity = min(args.tsne_perplexity, float(len(values) - 1))
        neighbor_count = min(args.umap_n_neighbors, len(values) - 1)
        coordinates_by_method = {
            "tsne": TSNE(n_components=2, perplexity=perplexity, init="pca",
                         learning_rate="auto", random_state=args.seed).fit_transform(values),
            "umap": UMAP(n_components=2, n_neighbors=neighbor_count,
                         min_dist=args.umap_min_dist, metric="cosine",
                         random_state=args.seed, n_jobs=1).fit_transform(values),
        }
        for method, coordinates in coordinates_by_method.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            for class_id, label in enumerate(labels):
                mask = targets == class_id
                if mask.any():
                    ax.scatter(coordinates[mask, 0], coordinates[mask, 1],
                               s=18, alpha=0.75, label=label)
            ax.set_title(f"{name} representation — {method.upper()}")
            ax.set_xlabel(f"{method.upper()} 1")
            ax.set_ylabel(f"{method.upper()} 2")
            ax.legend(title="Label", loc="best")
            fig.tight_layout()
            path = output_dir / f"{name}_{method}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            outputs[f"{name}_{method}"] = path
            coordinates_path = path.with_suffix(".csv")
            with coordinates_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "label", f"{method}_1", f"{method}_2"])
                for sample_id, class_id, xy in zip(selected_ids, targets, coordinates):
                    writer.writerow([sample_id, labels[int(class_id)], float(xy[0]), float(xy[1])])
            outputs[f"{name}_{method}_coordinates"] = coordinates_path
    return outputs
