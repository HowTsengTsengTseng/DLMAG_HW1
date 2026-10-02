"""Frozen Hugging Face MERT-v2 encoder and a trainable standardized MLP."""
import os

from dotenv import load_dotenv

# Set HF_HOME before importing Hugging Face, which reads cache settings at import time.
load_dotenv()

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoFeatureExtractor, AutoModel

from utils import MODEL_ID


class MERTEncoder(nn.Module):
    def __init__(self, model_id=MODEL_ID, revision="main"):
        super().__init__()
        kwargs = dict(revision=revision, token=os.getenv("HF_TOKEN") or None, trust_remote_code=True)
        self.processor = AutoFeatureExtractor.from_pretrained(model_id, **kwargs)
        self.backbone = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", **kwargs)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.hidden_size = self.backbone.config.hidden_size
        self.revision = getattr(self.backbone.config, "_commit_hash", None) or revision

    @torch.no_grad()
    def forward(self, waveforms):
        self.backbone.eval()
        inputs = self.processor(waveforms, sampling_rate=self.processor.sampling_rate,
                                padding=True, return_attention_mask=True, return_tensors="pt")
        inputs = inputs.to(next(self.backbone.parameters()).device)
        output = self.backbone(**inputs, return_dict=True)
        mask = output.feature_attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    @torch.no_grad()
    def fit_standardizer(self, training_features):
        self.feature_mean.copy_(training_features.mean(0))
        self.feature_std.copy_(training_features.std(0, unbiased=False).clamp_min(1e-6))

    def encode(self, features):
        standardized = (features - self.feature_mean) / self.feature_std
        normed = F.normalize(standardized, p=2, dim=-1)
        hidden = self.network[2](self.network[1](self.network[0](normed)))
        return F.normalize(hidden, p=2, dim=-1)

    @torch.no_grad()
    def set_prototypes(self, prototypes, temperature=0.1):
        """Set linear layer weights to temperature-scaled contrastive class prototypes."""
        normed_prototypes = F.normalize(prototypes, p=2, dim=-1)
        self.network[3].weight.copy_(normed_prototypes / temperature)
        self.network[3].bias.zero_()

    def forward(self, features, return_embedding=False):
        z = self.encode(features)
        logits = self.network[3](z)
        if return_embedding:
            return logits, z
        return logits


class ProjectionHead(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, h):
        return F.normalize(self.net(h.float()), dim=-1)


class LinearClassifier(nn.Module):
    def __init__(self, dim=1024, classes=6):
        super().__init__()
        self.linear = nn.Linear(dim, classes)

    def forward(self, x):
        return self.linear(x)


class NonLinearClassifier(nn.Module):
    def __init__(self, dim=1024, classes=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, classes))

    def forward(self, x):
        return self.net(x)


class CNNv2(nn.Module):
    '''ref: https://github.com/Ofir7909/music-classification-pytorch/blob/main/models.py'''
    name = "cnn-v2-mixed-dropout"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(p=0.25)

        # use dummy data to find the input size for the linear layer
        x = torch.randn(128, 128).view(-1, 1, 128, 128)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.drop4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
            print(f"{self._to_linear=}")

        return x

    def forward(self, x):
        x = self.convs(x)

        x = x.view(-1, self._to_linear)

        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class AudioSUCCNNv2(nn.Module):
    """Audio-SUC classifier that uses CNNv2 as its audio backbone.

    Inputs are mono waveforms shaped ``(batch, samples)``. Waveforms are
    converted to log-mel spectrograms, resized to CNNv2's expected 128x128
    input, and passed through its convolutional stack and 512-unit layer.
    The resulting representation feeds the era classifier and EC projection
    head described in Section 2.2 of He et al.
    """
    def __init__(self, num_classes=6, sample_rate=24000, seconds=30,
                 n_mels=224, n_fft=2048, hop_length=512, proj_dim=128):
        super().__init__()
        import torchaudio.transforms as T

        self.sample_rate = sample_rate
        self.target_samples = round(sample_rate * seconds)
        self.input_size = 128
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=2.0,
        )
        self.backbone = CNNv2()
        # CNNv2's original ten-way classifier is unused; fc1 is the 512-D h_a.
        del self.backbone.fc2
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, proj_dim),
        )

    def encode(self, waveforms):
        if waveforms.ndim != 2:
            raise ValueError(f"Expected mono waveforms shaped (batch, samples), got {tuple(waveforms.shape)}")
        if waveforms.shape[-1] < self.target_samples:
            waveforms = F.pad(waveforms, (0, self.target_samples - waveforms.shape[-1]))
        elif waveforms.shape[-1] > self.target_samples:
            waveforms = waveforms[..., :self.target_samples]
        mel = self.mel_extractor(waveforms)
        mel = torch.log(mel.clamp_min(1e-6)).unsqueeze(1)
        mel = F.interpolate(mel, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        x = self.backbone.convs(mel).flatten(start_dim=1)
        h = self.backbone.drop4(F.relu(self.backbone.fc1(x)))

        return h

    def embed(self, h):
        z = F.normalize(self.projection_head(h), p=2, dim=-1)
        return z
