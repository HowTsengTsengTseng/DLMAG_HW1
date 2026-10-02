"""Shared waveform augmentation composition used by feature extraction and training."""

from torchaudio_augmentations import (
    Compose,
    Delay,
    Gain,
    Noise,
    PitchShift,
    PolarityInversion,
    RandomApply,
    Reverb,
)


def make_audio_augmentation(sample_rate, seconds=30):
    """Build the project's standard stochastic waveform augmentation pipeline."""
    return Compose([
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
        RandomApply([Gain()], p=0.2),
        # HighLowPass(sample_rate=sample_rate),
        RandomApply([Delay(sample_rate=sample_rate)], p=0.5),
        RandomApply([PitchShift(
            n_samples=round(sample_rate * seconds),
            sample_rate=sample_rate,
        )], p=0.4),
        RandomApply([Reverb(sample_rate=sample_rate)], p=0.3),
    ])
