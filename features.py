"""Content-addressed frozen feature cache, shared by train and inference."""
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

from augmentations import make_audio_augmentation
from dataset import AudioDataset


def extract_features(rows, encoder, cache_dir, model_id, seconds=30, batch_size=1, with_transforms=False):
    if batch_size < 1 or not 0 < seconds <= 30:
        raise ValueError("batch_size must be positive and seconds must be in (0, 30]")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = make_audio_augmentation(encoder.processor.sampling_rate, seconds)

    dataset = AudioDataset(rows, encoder.processor.sampling_rate, seconds, transform if with_transforms else None)
    result = []
    pending = []
    for index, row in enumerate(tqdm(rows, desc="Reading feature cache")):
        digest = hashlib.sha256()
        with Path(row["path"]).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        spec = {"audio": digest.hexdigest(), "model": model_id, "revision": encoder.revision,
                "seconds": seconds, "sampling_rate": encoder.processor.sampling_rate,
                "pooling": "last-layer-masked-mean", "preprocess_version": 1}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        path = cache_dir / f"{key}.pt"
        if path.exists():
            feature = torch.load(path, map_location="cpu", weights_only=True)
            if feature.shape != (encoder.hidden_size,) or not torch.isfinite(feature).all():
                raise ValueError(f"Invalid cache; remove {path} and retry")
            result.append(feature)
        else:
            result.append(None)
            pending.append((index, path))
    for start in tqdm(range(0, len(pending), batch_size), desc="Extracting MERT features"):
        group = pending[start:start + batch_size]
        features = encoder([dataset[i] for i, _ in group]).detach().cpu().float()
        for (index, path), feature in zip(group, features):
            if not torch.isfinite(feature).all():
                raise ValueError(f"Non-finite MERT feature for {rows[index]['sample_id']}")
            temporary = path.with_suffix(".tmp")
            torch.save(feature, temporary)
            temporary.replace(path)
            result[index] = feature
    return torch.stack(result)
