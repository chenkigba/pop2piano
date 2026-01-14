"""Audio processing utilities for Pop2Piano."""

import numpy as np


def normalize(audio, min_y=-1.0, max_y=1.0, eps=1e-8):
    """Normalize audio to a given range.

    Args:
        audio: 1D audio array
        min_y: Minimum value
        max_y: Maximum value
        eps: Small value to prevent clipping

    Returns:
        Normalized audio array
    """
    assert len(audio.shape) == 1
    max_y -= eps
    min_y += eps
    amax = audio.max()
    amin = audio.min()
    audio = (max_y - min_y) * (audio - amin) / (amax - amin) + min_y
    return audio


def get_stereo(pop_y, midi_y, pop_scale=0.99):
    """Create stereo mix of original audio and MIDI synthesis.

    Args:
        pop_y: Original audio waveform
        midi_y: Synthesized MIDI waveform
        pop_scale: Scale factor for original audio

    Returns:
        Stereo audio array (2, samples)
    """
    if len(pop_y) > len(midi_y):
        midi_y = np.pad(midi_y, (0, len(pop_y) - len(midi_y)))
    elif len(pop_y) < len(midi_y):
        pop_y = np.pad(pop_y, (0, -len(pop_y) + len(midi_y)))
    stereo = np.stack((midi_y, pop_y * pop_scale))
    return stereo
