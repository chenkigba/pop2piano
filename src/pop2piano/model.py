"""High-level API for Pop2Piano."""

from typing import List, Optional, Union
from pathlib import Path

import torch
import librosa
from huggingface_hub import hf_hub_download

from pop2piano.config import get_default_config
from pop2piano.transformer import TransformerWrapper


class Pop2Piano:
    """High-level API for converting pop music to piano.

    Example:
        >>> from pop2piano import Pop2Piano
        >>> model = Pop2Piano(device="cuda")
        >>> midi = model.generate("song.mp3", composer="composer4")
        >>> midi.write("piano.mid")
    """

    MODEL_REPO = "sweetcocoa/pop2piano"
    MODEL_FILE = "model-1999-val_0.67311615.ckpt"

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize Pop2Piano model.

        Args:
            device: Device to run model on ("cuda" or "cpu")
            cache_dir: Directory to cache downloaded model
            checkpoint_path: Path to local checkpoint (skips download if provided)
        """
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._cache_dir = cache_dir
        self._checkpoint_path = checkpoint_path
        self._model: Optional[TransformerWrapper] = None
        self._config = None

    @property
    def config(self):
        """Get model configuration."""
        if self._config is None:
            self._config = get_default_config()
        return self._config

    @property
    def model(self) -> TransformerWrapper:
        """Get or lazily load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def composers(self) -> List[str]:
        """Get list of available composer styles."""
        return list(self.config.composer_to_feature_token.keys())

    def _load_model(self):
        """Load the model from checkpoint."""
        if self._checkpoint_path is not None:
            ckpt_path = self._checkpoint_path
        else:
            # Download from HuggingFace Hub
            ckpt_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILE,
                cache_dir=self._cache_dir,
            )

        self._model = TransformerWrapper.from_checkpoint(
            ckpt_path, self.config, device=self.device
        )
        self._model.eval()

    def generate(
        self,
        audio_path: Union[str, Path],
        composer: Optional[str] = None,
        n_bars: int = 2,
        steps_per_beat: int = 2,
    ):
        """Generate piano MIDI from audio file.

        Args:
            audio_path: Path to input audio file
            composer: Composer style ("composer1" to "composer21")
                     If None, randomly selects one
            n_bars: Number of bars per batch (default: 2)
            steps_per_beat: Steps per beat for quantization (default: 2)

        Returns:
            PrettyMIDI object with generated piano notes
        """
        audio_path = str(audio_path)
        return self.model.generate(
            audio_path=audio_path,
            composer=composer,
            n_bars=n_bars,
            steps_per_beat=steps_per_beat,
        )

    def generate_from_audio(
        self,
        audio: "np.ndarray",
        sample_rate: int,
        composer: Optional[str] = None,
        n_bars: int = 2,
        steps_per_beat: int = 2,
    ):
        """Generate piano MIDI from audio array.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            composer: Composer style ("composer1" to "composer21")
            n_bars: Number of bars per batch
            steps_per_beat: Steps per beat for quantization

        Returns:
            PrettyMIDI object with generated piano notes
        """
        return self.model.generate(
            audio_y=audio,
            audio_sr=sample_rate,
            composer=composer,
            n_bars=n_bars,
            steps_per_beat=steps_per_beat,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "Pop2Piano":
        """Create model from local checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run model on

        Returns:
            Pop2Piano instance
        """
        return cls(device=device, checkpoint_path=checkpoint_path)

    def to(self, device: str) -> "Pop2Piano":
        """Move model to specified device.

        Args:
            device: Target device ("cuda" or "cpu")

        Returns:
            self
        """
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self
