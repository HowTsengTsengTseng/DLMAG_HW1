# MERT Music Classification

This repository contains training and inference code for two six-class music
classification tasks: **Dataset A**, which predicts a recording's decade, and
**Dataset B**, which predicts its release market. The project includes frozen
MERT feature classifiers, contrastive and Audio-SUC experiments, and LoRA-based
MERT training workflows.

The repository contains source code and an example manifest only. Audio data,
model weights, feature caches, and trained checkpoints are not included.

## Project layout

```text
dataset.py                 Download data and prepare manifests
models.py                  MERT encoder, classifiers, and model utilities
features.py                MERT feature extraction and cache
loss.py                    Contrastive objectives
augmentations.py            Audio augmentation helpers
utils.py                   Labels, metrics, seeding, and plotting
train/
  train.py                 Frozen MERT features + MLP
  train_contrastive.py     Full-audio contrastive training and probing
  train_contrastive_crops.py 10-second crop contrastive training and probing
  train_audiosuc.py        Audio-SUC training workflow
  train_lora.py            10-second LoRA stage1, probe, and CE workflows
  train_lora_30s.py        30-second LoRA stage1 and probe workflows
inference/
  predict_mert.py          Frozen MERT MLP and contrastive checkpoints
  predict_lora.py          LoRA probe and CE checkpoints
  common.py                Shared test-manifest and submission helpers
infer_lora.py              LoRA prediction entry point
examples/manifest.csv      Manifest format example; contains no real samples
pyproject.toml, uv.lock    Project dependencies and lockfile
requirements.txt           Pip-compatible dependency list
```

## Setup

Python 3.11 or 3.12 is required. The project pins PyTorch 2.6.0,
torchaudio 2.6.0, and Transformers 4.53.2. Install the locked dependencies with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

Alternatively, install with pip:

```bash
pip install -r requirements.txt
```

The first MERT training or inference run downloads the model from Hugging Face.
Provide any required Hugging Face credentials in your environment; never commit
tokens or a populated `.env` file.

### Workaround for missing SoX on Linux workstations

On Linux systems without `libsox.so`, importing the PyPI version of
`torchaudio-augmentations` may fail while loading its `augment` dependency. If
you cannot install SoX, the following workaround edits the installed package
in the active virtual environment. These edits are local and may be overwritten
by `uv sync`; reapply them after syncing if needed.

Find the installed package directory:

```bash
uv run python - <<'PY'
import torchaudio_augmentations
print(torchaudio_augmentations.__path__[0])
PY
```

It is typically under `.venv/lib/python3.11/site-packages/torchaudio_augmentations`.
In `augmentations/pitch_shift.py`, replace the `augment`-based implementation
with `torch-pitch-shift`:

```python
import random

import torch
from torch_pitch_shift import get_fast_shifts, pitch_shift, semitones_to_ratio


class PitchShift:
    def __init__(self, n_samples, sample_rate, pitch_shift_min=-7.0, pitch_shift_max=7.0):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.fast_shifts = get_fast_shifts(
            sample_rate,
            lambda ratio: (
                semitones_to_ratio(pitch_shift_min)
                <= ratio <= semitones_to_ratio(pitch_shift_max)
                and ratio != 1
            ),
        )

    def process(self, audio):
        output = pitch_shift(
            input=audio.unsqueeze(0),
            shift=random.choice(self.fast_shifts),
            sample_rate=self.sample_rate,
            bins_per_octave=12,
        ).squeeze(0)
        if not torch.isfinite(output).all():
            return audio.clone()
        target_length = audio.shape[-1]
        if output.shape[-1] > target_length:
            return output[..., :target_length]
        if output.shape[-1] < target_length:
            padded = torch.zeros_like(audio)
            padded[..., :output.shape[-1]] = output
            return padded
        return output

    def __call__(self, audio):
        if audio.ndim == 3:
            return torch.stack([self.process(sample) for sample in audio], dim=0)
        return self.process(audio)
```

Then edit `augmentations/reverb.py`: replace its top-level `import augment` with
the guarded import below, and add the fallback at the start of `forward`:

```python
import torch

try:
    import augment
except (ImportError, OSError):
    augment = None
```

```python
def forward(self, audio):
    if augment is None:
        return audio.clone()
    # Keep the existing Reverb implementation below this guard.
```

With no SoX library, Reverb returns an unchanged copy of the audio while other
augmentations remain available. To check that Python imports the edited
`PitchShift` implementation:

```bash
uv run python - <<'PY'
import inspect
from torchaudio_augmentations import PitchShift
print(inspect.getfile(PitchShift))
PY
```


## Data and manifests

Download and extract the official datasets under `data/raw`:

```bash
uv run python dataset.py download --output data/raw
```

The downloader uses the assignment's [Google Drive folder](https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4).
The expected official manifests are `data/raw/dataset_A/manifest.csv` and
`data/raw/dataset_B/manifest.csv`, with audio files under each dataset's
`audio/` directory. Keep the official train, validation, and test assignments.

To export six canonical split manifests:

```bash
uv run python dataset.py prepare-all
```

This creates `data/manifests/{A,B}_{train,validation,test}.csv`. A manifest
contains `sample_id`, `path`, and `label` columns; paths are resolved relative
to `--data-root` (default `data/raw`). The labels are:

- Dataset A: `1960s`, `1970s`, `1980s`, `1990s`, `2000s`, `2010s`
- Dataset B: `US`, `UK`, `Brazil`, `Spain`, `Germany`, `Italy`

Training scripts accept the combined official manifest and filter by split, or
separate train and validation manifests. See each script's `--help` for options.

## Training

### Frozen MERT features and MLP

The default classifier extracts frozen MERT-v2 features, applies training-set
standardization and L2 normalization, then trains an MLP. The official combined
manifest is discovered under `data/raw` by default:

```bash
uv run python train/train.py --task A --output-dir runs/A
uv run python train/train.py --task B --output-dir runs/B
```

### Frozen-feature contrastive classifier

```bash
uv run python train/train_contrastive.py stage1 \
  --task A --output-dir runs/A_contrastive_stage1
uv run python train/train_contrastive.py probe \
  --task A --stage1 runs/A_contrastive_stage1 --probe-all \
  --output-dir runs/A_contrastive_probe
```

Stage1 saves the projection every five epochs and also saves the final epoch
when needed. Probe mode accepts one checkpoint, or probes every `epoch_*.pt`
checkpoint in a directory with `--probe-all`.

The full-audio workflow uses `train/train_contrastive.py`. The crop workflow
is kept in a separate script so both workflows remain available:

```bash
uv run python train/train_contrastive_crops.py stage1 \
  --task A --crop-seconds 10 \
  --output-dir runs/A_contrastive_10s_stage1
uv run python train/train_contrastive_crops.py probe \
  --task A --stage1 runs/A_contrastive_10s_stage1 --probe-all \
  --output-dir runs/A_contrastive_10s_probe
```

### Audio-SUC

```bash
uv run python train/train_audiosuc.py --task A --output-dir runs/A_audiosuc
uv run python train/train_audiosuc.py --task B --output-dir runs/B_audiosuc
```

### LoRA workflows

The 10-second workflow supports supervised-contrastive adapter training
(`stage1`), a linear probe (`probe`), and a joint LoRA + cross-entropy baseline
(`ce`). For example:

```bash
uv run python train/train_lora.py stage1 --task A --output-dir runs/A_lora_s1
uv run python train/train_lora.py probe --task A \
  --stage1 runs/A_lora_s1/epoch_010.pt --output-dir runs/A_lora_probe
uv run python train/train_lora.py ce --task A --output-dir runs/A_lora_ce
```

The separate 30-second workflow provides `stage1` and `probe` modes:

```bash
uv run python train/train_lora_30s.py stage1 --task A --output-dir runs/A_lora_30s_s1
uv run python train/train_lora_30s.py probe --task A \
  --stage1 runs/A_lora_30s_s1 --output-dir runs/A_lora_30s_probe
```

Repeat any example with `--task B` for Dataset B. Training outputs are written
to the requested run directory; use a new directory for each run.

## Test-set inference

Both inference scripts produce the assignment submission structure: a mapping
from each test sample ID to its three highest-ranked labels. By default, test
manifests are discovered from `data/raw/dataset_{A,B}/manifest.csv` or
`data/manifests/{A,B}_test.csv` and filtered to the test split. Pass
`--manifest-a` and `--manifest-b` to select other files.

### Frozen MERT MLP or contrastive checkpoints

`inference/predict_mert.py` supports checkpoints from `train/train.py` and
`train/train_contrastive.py`:

```bash
uv run python inference/predict_mert.py \
  --checkpoint-a runs/A/best.pt \
  --checkpoint-b runs/B/best.pt \
  --output predictions/mert.json
```

### LoRA probe or CE checkpoints

`inference/predict_lora.py` supports classifier checkpoints from the LoRA
training scripts:

```bash
uv run python inference/predict_lora.py \
  --checkpoint-a runs/A_lora_probe/best.pt \
  --checkpoint-b runs/B_lora_probe/best.pt \
  --output predictions/lora.json
```

The 10-second LoRA workflow uses three fixed crops and averages their
probabilities; the 30-second probe uses one full recording per sample. A LoRA
`stage1` checkpoint has no classifier, so run `probe` or `ce` before prediction.

### Mixed-model inference

`inference/predict.py` is the unified inference entry point. Task A and Task B
may use different checkpoint families; the checkpoint `kind` selects the
appropriate LoRA or frozen-MERT pipeline automatically:

```bash
uv run python inference/predict.py \
  --checkpoint-a runs/A_lora_probe/best.pt \
  --checkpoint-b runs/B/best.pt \
  --output predictions/mixed.json
```

Use `--manifest-a` or `--manifest-b` when a task uses a non-default test
manifest. The output keeps both `dataset_A` and `dataset_B` sections.

### Validation metrics

Use `inference/validate.py` with the same per-task checkpoints to compute
recording-level cross-entropy loss, top-1 accuracy, and top-3 accuracy on the
validation split:

```bash
uv run python inference/validate.py \
  --checkpoint-a runs/A_lora_probe/best.pt \
  --checkpoint-b runs/B/best.pt \
  --output validation/mixed.json
```

Validation manifests are discovered from the task manifest and filtered to the
`validation` split. Pass `--manifest-a` or `--manifest-b` to override them.
For crop checkpoints, crop probabilities are combined before all metrics are
calculated.

Both prediction scripts can validate their output against an optional
assignment template using `--template`. The template is used to check sample
IDs; its example rankings are not used as predictions. Choose the inference
entry point and checkpoint type that match the model being submitted.

## Model and evaluation notes

The frozen-feature MLP path converts audio to mono, resamples to 24 kHz, and
uses up to 30 seconds of audio with the frozen MERT-v2 encoder. Its default
head is `Linear(1024, 256)`, GELU, dropout, and a six-class output layer.
Training-set statistics are stored with the checkpoint. Validation is used for
model selection; test labels are not used for training or selection.

MERT is a large model, and feature extraction can require substantial compute
and memory. The frozen-feature scripts default to automatic CPU/CUDA selection
and extraction batch size 1. CUDA memory requirements have not been measured in
this repository.

## References

- [MERT-v2-30s model card](https://huggingface.co/m-a-p/MERT-v2-30s)
- [MERT research repository](https://github.com/yizhilll/MERT)
- [uv documentation](https://docs.astral.sh/uv/)
