"""Train Audio-SUC model using Supervised Contrastive Learning (Section 2.2, He et al., 2024).

Objective:
    L = L_MLE + beta * L_EC
where:
    L_MLE: Maximum Likelihood Estimation (Cross-Entropy loss) on classification head f(h_a)
    L_EC:  Era Contrastive loss on projection head g_theta(h_a) = z (Eq. 3 in paper)
"""
import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path when running from subfolder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import check_disjoint, read_manifest
from features import extract_features
from models import AudioSUC, MERTEncoder
from utils import LABELS, MODEL_ID, get_device, metrics, save_confusion, save_json, seed_everything


class EraContrastiveLoss(nn.Module):
    """Supervised Era Contrastive (EC) Loss from Eq. 3 in He et al. (2024) / Khosla et al. (2020).

    Forces clusters of audio embeddings belonging to the same era class to be pulled together
    in the embedding space, while pushing apart audio embeddings from different era classes.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # z on unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Subtract max for numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(sim) - torch.eye(batch_size, device=embeddings.device)

        # Mask for positive pairs (same era class, excluding self)
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * logits_mask

        # Log-probability: log [ exp(sim_ij / tau) / sum_{k != i} exp(sim_ik / tau) ]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))

        # Mean over positive pairs for each anchor with at least one positive
        pos_count = label_mask.sum(1)
        valid_anchors = pos_count > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        mean_log_prob_pos = (label_mask * log_prob).sum(1) / pos_count.clamp_min(1.0)
        loss = -mean_log_prob_pos[valid_anchors].mean()
        return loss


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimension of audio representation h_a")
    parser.add_argument("--proj-dim", type=int, default=128, help="Dimension of contrastive projection head z")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    # Audio-SUC hyperparameters (beta and tau)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight beta for Era Contrastive loss L_EC in L = L_MLE + beta * L_EC (default: 0.5)")
    parser.add_argument("--temperature", "--tau", type=float, default=0.1,
                        help="Temperature tau for Era Contrastive loss (default: 0.1)")
    parser.add_argument("--use-audio-cnn", action="store_true",
                        help="Train 2D CNN directly on mel-spectrograms extracted from raw audio as in the paper (bypasses MERT)")
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

    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim, args.proj_dim) < 1:
        parser.error("epochs, patience, batch-size, hidden-dim and proj-dim must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid optimizer or dropout parameters")
    if args.beta < 0 or args.temperature <= 0:
        parser.error("beta must be non-negative and temperature must be positive")

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

    y_train = torch.tensor([labels.index(r["label"]) for r in train_rows])
    y_val = torch.tensor([labels.index(r["label"]) for r in val_rows])

    if args.use_audio_cnn:
        from dataset import AudioDataset
        ds_train = AudioDataset(train_rows, sampling_rate=24000, seconds=args.seconds)
        ds_val = AudioDataset(val_rows, sampling_rate=24000, seconds=args.seconds)

        class AudioLabelDataset(torch.utils.data.Dataset):
            def __init__(self, audio_ds, labels_tensor):
                self.audio_ds, self.labels = audio_ds, labels_tensor
            def __len__(self):
                return len(self.audio_ds)
            def __getitem__(self, idx):
                return torch.from_numpy(self.audio_ds[idx]), self.labels[idx]

        loader = DataLoader(AudioLabelDataset(ds_train, y_train), batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(AudioLabelDataset(ds_val, y_val), batch_size=args.batch_size, shuffle=False)
        model_args = dict(
            num_classes=num_classes,
            proj_dim=args.proj_dim,
            sample_rate=24000,
            n_mels=224,
            n_fft=2048,
            hop_length=512,
        )
        model = AudioSUC(**model_args).to(device)
        revision = "audio_cnn"
    else:
        encoder = MERTEncoder(args.model_id, args.revision).to(device)
        revision, input_dim = encoder.revision, encoder.hidden_size
        features = [extract_features(rows, encoder, args.cache_dir, args.model_id,
                                     args.seconds, args.extract_batch_size) for rows in (train_rows, val_rows)]
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

        x_train, x_val = features
        model_args = dict(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        )
        model = AudioSUC(**model_args)
        model.fit_standardizer(x_train)
        model.to(device)

        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    criterion_ec = EraContrastiveLoss(temperature=args.temperature)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss, stale, history = float("inf"), 0, []
    save_json(output / "config.json", vars(args) | {
        "resolved_revision": revision,
        "labels": labels,
        "model_architecture": "AudioSUC",
        "loss_formula": "L_MLE + beta * L_EC",
    })

    print(f"Training Audio-SUC (beta={args.beta}, tau={args.temperature})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_mle, total_ec = 0.0, 0.0, 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, z = model(x)  # returns logits and normalized projection z
            loss_mle = F.cross_entropy(logits, y)

            if args.beta > 0:
                loss_ec = criterion_ec(z, y)
                loss = loss_mle + args.beta * loss_ec
            else:
                loss_ec = torch.tensor(0.0)
                loss = loss_mle

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_mle += loss_mle.item() * len(y)
            total_ec += loss_ec.item() * len(y)

        # Validation phase: uses classification head f(h_a)
        model.eval()
        val_loss_total = 0.0
        all_logits = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vlogits = model(vx)
                vloss = F.cross_entropy(vlogits, vy)
                val_loss_total += vloss.item() * len(vy)
                all_logits.append(vlogits.cpu())
            logits = torch.cat(all_logits, dim=0)
            val_loss = val_loss_total / len(y_val)

        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError("Non-finite validation loss")

        scores = metrics(logits, y_val, labels)
        n_train = len(y_train)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / n_train,
            "train_mle_loss": total_mle / n_train,
            "train_ec_loss": total_ec / n_train,
            "val_loss": val_loss,
            "top1": scores["top1"],
            "top3": scores["top3"],
        }
        history.append(record)
        print(f"Epoch {epoch:03d} | Train: {record['train_loss']:.4f} (MLE: {record['train_mle_loss']:.4f}, EC: {record['train_ec_loss']:.4f}) | Val Loss: {val_loss:.4f} | Top-1: {scores['top1']*100:.2f}% | Top-3: {scores['top3']*100:.2f}%", flush=True)

        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            checkpoint = {
                "format_version": 1,
                "model_class": "AudioSUC",
                "task": args.task,
                "labels": labels,
                "model_id": args.model_id,
                "revision": revision,
                "seconds": args.seconds,
                "model_args": model_args,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "seed": args.seed,
                "train_ids": [r["sample_id"] for r in train_rows],
                "validation_ids": [r["sample_id"] for r in val_rows],
                "beta": args.beta,
                "temperature": args.temperature,
            }
            torch.save(checkpoint, output / "best.pt")
            save_json(output / "validation_metrics.json", scores | {"epoch": epoch, "loss": val_loss})
            save_confusion(scores, output / "validation_confusion.png")
        else:
            stale += 1

        save_json(output / "history.json", history)
        if stale >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"Best checkpoint saved to: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
