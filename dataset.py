"""Explicit manifests preserve the official splits; never infer labels from test data."""
import argparse
import csv
from pathlib import Path
import stat
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from utils import LABELS

DRIVE_URL = "https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4"


def resolve_audio_path(root, manifest_path, raw_path, task):
    raw_p = Path(raw_path)
    if raw_p.is_absolute() and raw_p.is_file():
        return raw_p.resolve()
    candidates = [
        Path(root) / raw_p,
        Path(root) / f"dataset_{task}" / raw_p,
        Path(manifest_path).parent / raw_p,
        Path(manifest_path).parent / f"dataset_{task}" / raw_p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (Path(root) / raw_p).resolve()


def read_manifest(path, root, task, split):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        path_col = None
        for candidate in ["path", "audio_path"]:
            if candidate in fields:
                path_col = candidate
                break
        if "sample_id" not in fields or path_col is None:
            raise ValueError(f"{path}: required columns: sample_id and either path or audio_path")
        has_split_col = "split" in fields
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"Empty manifest: {path}")

    rows = []
    ids, paths = set(), set()
    for row in raw_rows:
        if has_split_col and row["split"].strip() != split:
            continue
        sid = row["sample_id"].strip()
        audio = resolve_audio_path(root, path, row[path_col].strip(), task)
        label = (row.get("label") or "").strip()
        if not sid or sid in ids or audio in paths:
            raise ValueError(f"Empty/duplicate ID or duplicate audio path: {sid}")
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio}")
        if split != "test" and label not in LABELS[task]:
            raise ValueError(f"Invalid {task} label {label!r} for {sid}; expected {LABELS[task]}")
        row.update(sample_id=sid, path=str(audio), label=label)
        ids.add(sid)
        paths.add(audio)
        rows.append(row)

    if not rows:
        raise ValueError(f"No rows matching split {split!r} in manifest: {path}")
    return rows


def check_disjoint(*splits):
    ids, paths = set(), set()
    for rows in splits:
        current_ids = {r["sample_id"] for r in rows}
        current_paths = {r["path"] for r in rows}
        if ids & current_ids or paths & current_paths:
            raise ValueError("Official splits overlap in sample IDs or audio paths")
        ids |= current_ids
        paths |= current_paths


class AudioDataset(Dataset):
    def __init__(self, rows, sampling_rate=24000, seconds=30.0, transform=None):
        self.rows, self.sampling_rate, self.seconds, self.transform = rows, sampling_rate, seconds, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, sr = sf.read(row["path"], dtype="float32", always_2d=True)
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Empty or non-finite audio: {row['path']}")
        # Keep the centered excerpt, deterministic for both training and inference.
        size = min(len(audio), round(self.seconds * sr))
        start = (len(audio) - size) // 2
        wave = torch.from_numpy(audio[start:start + size].mean(axis=1))
        if sr != self.sampling_rate:
            wave = AF.resample(wave, sr, self.sampling_rate)
        if len(wave) < 1025:
            raise ValueError(f"Audio shorter than MERT-v2 minimum 1025 samples: {row['path']}")

        if self.transform:
            wave = self.transform(wave)

        return wave.numpy()


class LabeledAudioDataset(Dataset):
    """Pair lazily loaded waveforms with their encoded era labels."""
    def __init__(self, audio_dataset, labels):
        self.audio_dataset = audio_dataset
        self.labels = labels

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, index):
        return torch.as_tensor(self.audio_dataset[index], dtype=torch.float32), self.labels[index]


class TwoViewDataset(Dataset):
    def __init__(self, rows, sample_rate=24000, crop_seconds=10):
        self.rows, self.crop_samples = rows, round(sample_rate * crop_seconds)
        self.base = AudioDataset(rows, sample_rate, seconds=30)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        wave = torch.from_numpy(self.base[index])
        views = []
        for _ in range(2):
            if len(wave) > self.crop_samples:
                start = random.randint(0, len(wave) - self.crop_samples)
                views.append(wave[start:start + self.crop_samples].numpy())
            else:
                views.append(F.pad(wave, (0, self.crop_samples - len(wave))).numpy())
        return views[0], views[1], int(self.rows[index]["label_index"])


class FullAudioTwoViewDataset(Dataset):
    """Load a full 30-second recording and independently augment two views."""

    def __init__(self, rows, sample_rate=24000, seconds=30.0, transform=None):
        if not 0 < seconds <= 30:
            raise ValueError("seconds must be in (0, 30]")
        self.rows = rows
        self.sample_rate = sample_rate
        self.target_samples = round(sample_rate * seconds)
        self.audio = AudioDataset(rows, sample_rate, seconds=seconds)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def _view(self, wave):
        wave = wave[:self.target_samples]
        if wave.numel() < self.target_samples:
            wave = torch.nn.functional.pad(wave, (0, self.target_samples - wave.numel()))
        # torchaudio-augmentations expects [channels, samples]. Crop/pad after
        # transforms too, since some reverb implementations append a tail.
        view = wave.unsqueeze(0)
        if self.transform is not None:
            view = self.transform(view.clone())
        view = view.reshape(-1)[:self.target_samples]
        if view.numel() < self.target_samples:
            view = torch.nn.functional.pad(view, (0, self.target_samples - view.numel()))
        if not torch.isfinite(view).all():
            raise ValueError("Audio augmentation produced non-finite samples")
        return view

    def __getitem__(self, index):
        wave = torch.as_tensor(self.audio[index], dtype=torch.float32)
        view1 = self._view(wave)
        view2 = self._view(wave)
        return view1, view2, int(self.rows[index]["label_index"])


class ClassBalancedBatchSampler:
    def __init__(self, rows, classes=6, recordings_per_class=2, batches=0):
        self.by_class = {c: [i for i, r in enumerate(rows) if r["label_index"] == c]
                         for c in range(classes)}
        self.classes, self.rpc = classes, recordings_per_class
        self.batches = batches or max(1, len(rows) // (classes * recordings_per_class))

    def __iter__(self):
        for _ in range(self.batches):
            chosen = []
            for c in random.sample(range(self.classes), self.classes):
                pool = self.by_class[c]
                chosen.extend(random.choices(pool, k=self.rpc) if len(pool) < self.rpc
                               else random.sample(pool, self.rpc))
            yield chosen

    def __len__(self): return self.batches


def download_dataset(output):
    import gdown
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(url=DRIVE_URL, output=str(output), quiet=False, use_cookies=False, remaining_ok=False)
    if not files:
        raise RuntimeError("Google Drive download failed; check sharing permissions/quota")
    # Extract only regular files, with traversal and symlink protection.
    for path in map(Path, files):
        if path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = (output / info.filename).resolve()
                if not target.is_relative_to(output.resolve()) or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unsafe archive entry: {info.filename}")
            archive.extractall(output)
    print(f"Downloaded and extracted into {output.resolve()}. Keep official split assignments when preparing manifests.")


def convert_manifest(args):
    """Adapt official CSV column names without guessing metadata or splitting samples."""
    with Path(args.source).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = [args.id_column, args.path_column]
        if args.split != "test":
            required.append(args.label_column)
        if args.split_column:
            required.append(args.split_column)
        if not set(required).issubset(fields):
            raise ValueError(f"Missing columns {set(required) - set(fields)}; found {fields}")
        rows = []
        for row in reader:
            if args.split_column and row[args.split_column] != (args.split_value or args.split):
                continue
            rows.append({"sample_id": row[args.id_column], "path": row[args.path_column],
                         "label": row.get(args.label_column, "") if args.split != "test" else ""})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    read_manifest(destination, args.data_root, args.task, args.split)
    print(f"Saved {len(rows)} rows to {destination}")


def prepare_all_manifests(data_root="data/raw", output_dir="data/manifests"):
    """Export canonical manifests for all tasks and splits from raw dataset manifests."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for task in ["A", "B"]:
        source = data_root / f"dataset_{task}" / "manifest.csv"
        if not source.exists():
            print(f"Skipping task {task}: {source} not found")
            continue
        with source.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            path_col = "path" if "path" in fields else "audio_path"
            rows = list(reader)
        for split in ["train", "validation", "test"]:
            split_rows = []
            for r in rows:
                if r.get("split", "").strip() != split:
                    continue
                audio_p = r[path_col].strip()
                rel_path = f"dataset_{task}/{audio_p}" if not audio_p.startswith(f"dataset_{task}/") else audio_p
                label = r.get("label", "").strip() if split != "test" else ""
                split_rows.append({"sample_id": r["sample_id"].strip(), "path": rel_path, "label": label})
            dest = output_dir / f"{task}_{split}.csv"
            with dest.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label"])
                writer.writeheader()
                writer.writerows(split_rows)
            read_manifest(dest, data_root, task, split)
            generated.append(dest)
            print(f"Generated {dest} ({len(split_rows)} rows)")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--output", default="data/raw")
    convert = sub.add_parser("prepare", help="Convert an official CSV into a canonical manifest")
    convert.add_argument("--source", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--data-root", required=True)
    convert.add_argument("--task", choices=LABELS, required=True)
    convert.add_argument("--split", choices=["train", "validation", "test"], required=True)
    convert.add_argument("--id-column", default="sample_id")
    convert.add_argument("--path-column", default="path")
    convert.add_argument("--label-column", default="label")
    convert.add_argument("--split-column")
    convert.add_argument("--split-value")
    prep_all = sub.add_parser("prepare-all", help="Generate all 6 canonical manifests from data/raw")
    prep_all.add_argument("--data-root", default="data/raw")
    prep_all.add_argument("--output-dir", default="data/manifests")
    args = parser.parse_args()
    if args.command == "download":
        download_dataset(args.output)
    elif args.command == "prepare-all":
        prepare_all_manifests(args.data_root, args.output_dir)
    else:
        convert_manifest(args)


if __name__ == "__main__":
    main()
