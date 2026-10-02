from dataset import AudioDataset, LabeledAudioDataset
import torch
from torch.utils.data import DataLoader
from utils import collate_audio


def stratified_indices(labels, max_samples, seed):
    """Select a deterministic, approximately class-balanced t-SNE subset."""
    if len(labels) <= max_samples:
        return torch.arange(len(labels))

    generator = torch.Generator().manual_seed(seed)
    classes = torch.unique(labels, sorted=True)
    quota = max_samples // len(classes)
    selected, remaining = [], []
    for class_id in classes:
        indices = torch.where(labels == class_id)[0]
        order = torch.randperm(len(indices), generator=generator)
        take = min(quota, len(indices))
        selected.append(indices[order[:take]])
        remaining.append(indices[order[take:]])
    selected = torch.cat(selected)
    leftover = max_samples - len(selected)
    if leftover:
        remaining = torch.cat(remaining)
        order = torch.randperm(len(remaining), generator=generator)
        selected = torch.cat((selected, remaining[order[:leftover]]))
    return selected.sort().values


@torch.no_grad()
def save_embedding_visualizations(model, rows, labels_tensor, labels, args, device, output_dir, projection=None):
    """Save t-SNE and UMAP plots of best-model EC embeddings, colored by class."""
    from sklearn.manifold import TSNE
    from umap import UMAP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = stratified_indices(labels_tensor, args.tsne_max_samples, args.seed)
    selected_rows = [rows[index] for index in indices.tolist()]
    selected_labels = labels_tensor[indices]
    dataset = LabeledAudioDataset(
        AudioDataset(selected_rows, sampling_rate=args.sample_rate, seconds=args.seconds),
        selected_labels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_audio)

    model.eval()
    embeddings = []
    for waveforms, _ in loader:
        z = model.embed(model.encode(waveforms.to(device)))
        embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings).numpy()
    targets = selected_labels.numpy()

    if len(embeddings) < 3:
        raise ValueError("Need at least three examples to compute t-SNE")
    perplexity = min(args.tsne_perplexity, float(len(embeddings) - 1))
    tsne_coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(embeddings)

    umap_neighbors = min(args.umap_n_neighbors, len(embeddings) - 1)
    umap_coordinates = UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
        n_jobs=1,
    ).fit_transform(embeddings)

    def save_plot(coordinates, method, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        for class_id, label in enumerate(labels):
            mask = targets == class_id
            if mask.any():
                ax.scatter(coordinates[mask, 0], coordinates[mask, 1],
                           s=18, alpha=0.75, label=label)
        ax.set_title(f"Audio-SUC CNNv2 EC embeddings ({method})")
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.legend(title="Label", loc="best")
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    return (
        save_plot(tsne_coordinates, "t-SNE", "embedding_tsne.png"),
        save_plot(umap_coordinates, "UMAP", "embedding_umap.png"),
    )
