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
