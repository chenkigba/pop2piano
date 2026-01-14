"""Beat detection using librosa (cross-platform replacement for essentia)."""

import librosa
import numpy as np
import scipy.interpolate as interp

SAMPLERATE = 44100


def interpolate_beat_times(beat_times, steps_per_beat, extend=False):
    """Interpolate between beat times."""
    beat_times_function = interp.interp1d(
        np.arange(beat_times.size),
        beat_times,
        bounds_error=False,
        fill_value="extrapolate",
    )
    if extend:
        beat_steps_8th = beat_times_function(
            np.linspace(0, beat_times.size, beat_times.size * steps_per_beat + 1)
        )
    else:
        beat_steps_8th = beat_times_function(
            np.linspace(0, beat_times.size - 1, beat_times.size * steps_per_beat - 1)
        )
    return beat_steps_8th


def extract_rhythm(song, y=None):
    """Extract rhythm information using librosa.

    Args:
        song: Path to audio file
        y: Audio waveform array (optional)

    Returns:
        bpm: Estimated BPM
        beat_times: Array of beat time points
        confidence: Confidence score (always 1.0 for librosa)
        estimates: BPM estimates array
        beat_intervals: Beat intervals
    """
    if y is None:
        y, sr = librosa.load(song, sr=SAMPLERATE)
    else:
        sr = SAMPLERATE

    # Use librosa's beat detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Handle tempo being an array
    if isinstance(tempo, np.ndarray):
        bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        bpm = float(tempo)

    # Convert frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Ensure beat_times is numpy array with float32
    beat_times = np.array(beat_times, dtype=np.float32)

    # Calculate beat intervals
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
    else:
        beat_intervals = np.array([60.0 / bpm], dtype=np.float32)

    # Return in essentia-compatible format
    confidence = 1.0
    estimates = np.array([bpm], dtype=np.float32)

    return bpm, beat_times, confidence, estimates, beat_intervals
