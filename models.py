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


class SVMClassifier(nn.Module):
    """Linear Support Vector Machine classifier with integrated feature standardizer.

    Standardizes features, L2-normalizes, and computes the linear SVM decision margin:
        decision_scores = linear(normalize((x - mean) / std))
    Fitted via scikit-learn (LinearSVC/SVC) and stored as PyTorch parameters for seamless
    GPU/CPU inference in test.py without scikit-learn dependencies.
    """
    def __init__(self, input_dim=1024, num_classes=6):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, num_classes)

    @torch.no_grad()
    def fit_standardizer(self, training_features):
        self.feature_mean.copy_(training_features.mean(0))
        self.feature_std.copy_(training_features.std(0, unbiased=False).clamp_min(1e-6))

    @torch.no_grad()
    def set_weights(self, coef, intercept):
        """Set linear weights and bias from fitted sklearn model."""
        import numpy as np
        if isinstance(coef, np.ndarray):
            coef = torch.from_numpy(coef)
        if isinstance(intercept, np.ndarray):
            intercept = torch.from_numpy(intercept)
        self.linear.weight.copy_(coef.float())
        self.linear.bias.copy_(intercept.float())

    def forward(self, features):
        standardized = (features - self.feature_mean) / self.feature_std
        normed = F.normalize(standardized, p=2, dim=-1)
        return self.linear(normed)


class AudioCNNBackbone(nn.Module):
    """Stack of CNN layers from Section 2.1 (He et al., 2024 / Ibrahim et al., 2020).

    Computes: H_l = ELU(BN(CNN(H_{l-1}))) with 3x3 kernels followed by average pooling.
    """
    def __init__(self, in_channels=1, channels=(32, 64, 128, 256, 256)):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ])
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        feat = self.conv(x)
        return feat.mean(dim=[-2, -1])  # Global average pooling -> (batch, out_dim)


class AudioCNN(nn.Module):
    """Audio-CNN baseline model from Section 2.1 (He et al., 2024).

    Mel-spectrogram -> AudioCNNBackbone -> Classifier f(h_a).
    Trained with cross-entropy loss L_MLE.
    """
    def __init__(self, num_classes=6, in_channels=1, channels=(32, 64, 128, 256, 256),
                 sample_rate=24000, n_mels=224, n_fft=2048, hop_length=512):
        super().__init__()
        import torchaudio.transforms as T
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        self.backbone = AudioCNNBackbone(in_channels=in_channels, channels=channels)
        self.classifier = nn.Linear(self.backbone.out_dim, num_classes)

    def extract_mel(self, audio):
        # audio: (B, T)
        mel = self.mel_extractor(audio)
        return torch.log(mel.clamp_min(1e-6)).unsqueeze(1)

    def forward(self, x):
        if x.dim() == 2:  # raw audio waveform (B, T)
            x = self.extract_mel(x)
        elif x.dim() == 3:  # mel-spectrogram (B, n_mels, time)
            x = x.unsqueeze(1)
        h_a = self.backbone(x)
        return self.classifier(h_a)


class AudioSUC(nn.Module):
    """Audio-SUC: Supervised Contrastive Learning with CNN backbone (Section 2.2, He et al., 2024).

    Input: Mel-spectrogram x (or raw audio waveform, or 1D feature tensor).
    Backbone: CNN layers computing audio representation h_a.
    Classification Head: f(h_a) -> logits (trained with L_MLE / Cross-Entropy).
    Projection Head: g_theta(h_a) -> z on unit hypersphere (trained with L_EC / Era Contrastive Loss).

    Objective:
      L = L_MLE + beta * L_EC
    """
    def __init__(self, num_classes=6, in_channels=1, channels=(32, 64, 128, 256, 256),
                 proj_dim=128, sample_rate=24000, n_mels=224, n_fft=2048, hop_length=512,
                 input_dim=None, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim

        # 1D CNN adapter for pre-extracted 1D embeddings (e.g. MERT features)
        if input_dim is not None:
            self.register_buffer("feature_mean", torch.zeros(input_dim))
            self.register_buffer("feature_std", torch.ones(input_dim))
            self.backbone_1d = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.AvgPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.AvgPool1d(2),
                nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            backbone_dim = hidden_dim
        else:
            backbone_dim = channels[-1]

        # 2D CNN MelSpectrogram frontend (Section 3.1)
        import torchaudio.transforms as T
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        self.backbone_2d = AudioCNNBackbone(in_channels=in_channels, channels=channels)

        # Classification head f(h_a) -> logits
        self.classifier = nn.Linear(backbone_dim, num_classes)

        # Projection head g_theta(h_a) -> z on unit hypersphere
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ELU(),
            nn.Linear(backbone_dim, proj_dim),
        )

    @torch.no_grad()
    def fit_standardizer(self, training_features):
        if hasattr(self, "feature_mean"):
            self.feature_mean.copy_(training_features.mean(0))
            self.feature_std.copy_(training_features.std(0, unbiased=False).clamp_min(1e-6))

    def extract_mel(self, audio):
        # audio: (B, T)
        mel = self.mel_extractor(audio)
        return torch.log(mel.clamp_min(1e-6)).unsqueeze(1)  # (B, 1, n_mels, time)

    def encode(self, x):
        """Extract audio embedding h_a and normalized contrastive projection z."""
        if x.dim() == 2 and hasattr(self, "feature_mean") and x.shape[1] == self.input_dim:
            # 1D feature tensor (e.g. MERT features) -> 1D CNN
            standardized = (x - self.feature_mean) / self.feature_std
            normed = F.normalize(standardized, p=2, dim=-1).unsqueeze(1)  # (B, 1, D)
            h_a = self.backbone_1d(normed).squeeze(-1)  # (B, hidden_dim)
        elif x.dim() == 2:
            # Raw audio waveform (B, T) -> Mel-spectrogram -> 2D CNN
            mel = self.extract_mel(x)
            h_a = self.backbone_2d(mel)
        elif x.dim() == 3:
            # (B, n_mels, time) -> 2D CNN
            h_a = self.backbone_2d(x.unsqueeze(1))
        else:
            # (B, 1, n_mels, time) -> 2D CNN
            h_a = self.backbone_2d(x)

        z = F.normalize(self.projection_head(h_a), p=2, dim=-1)
        return h_a, z

    def forward(self, x, return_projection=False):
        h_a, z = self.encode(x)
        logits = self.classifier(h_a)
        if return_projection or self.training:
            return logits, z
        return logits


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
