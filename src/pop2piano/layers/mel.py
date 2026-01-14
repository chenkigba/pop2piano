"""Mel spectrogram layers for Pop2Piano."""

import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    """Convert audio waveform to log mel spectrogram."""

    def __init__(self) -> None:
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=4096,
            hop_length=1024,
            f_min=10.0,
            n_mels=512,
        )

    def forward(self, x):
        """Convert audio to log mel spectrogram.

        Args:
            x: Audio tensor (batch, sample)

        Returns:
            Log mel spectrogram (batch, freq, frame)
        """
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                X = self.melspectrogram(x)
                X = X.clamp(min=1e-6).log()

        return X


class ConcatEmbeddingToMel(nn.Module):
    """Concatenate composer embedding to mel spectrogram."""

    def __init__(self, embedding_offset, n_vocab, n_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim)
        self.embedding_offset = embedding_offset

    def forward(self, feature, index_value):
        """Concatenate embedding to feature.

        Args:
            index_value: Composer index (batch,)
            feature: Mel features (batch, time, feature_dim)

        Returns:
            Combined features (batch, 1 + time, feature_dim)
        """
        index_shifted = index_value - self.embedding_offset

        # (batch, 1, feature_dim)
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        # (batch, 1 + time, feature_dim)
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds
