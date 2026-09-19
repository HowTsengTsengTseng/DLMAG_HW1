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


def read_manifest(path, root, task, split):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"sample_id", "path"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: sample_id,path,label (label optional for test)")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids, paths = set(), set()
    for row in rows:
        sid = row["sample_id"].strip()
        audio = (Path(root) / row["path"]).resolve()
        label = (row.get("label") or "").strip()
        if not sid or sid in ids or audio in paths:
            raise ValueError(f"Empty/duplicate ID or duplicate audio path: {sid}")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if split != "test" and label not in LABELS[task]:
            raise ValueError(f"Invalid {task} label {label!r} for {sid}; expected {LABELS[task]}")
        row.update(sample_id=sid, path=str(audio), label=label)
        ids.add(sid)
        paths.add(audio)
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
    def __init__(self, rows, sampling_rate=24000, seconds=30.0):
        self.rows, self.sampling_rate, self.seconds = rows, sampling_rate, seconds

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
        return wave.numpy()


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
    args = parser.parse_args()
    if args.command == "download":
        download_dataset(args.output)
    else:
        convert_manifest(args)


if __name__ == "__main__":
    main()
