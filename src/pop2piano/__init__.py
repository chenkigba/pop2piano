"""Pop2Piano: Convert pop music to piano cover using AI.

Example:
    >>> from pop2piano import Pop2Piano
    >>> model = Pop2Piano(device="cuda")
    >>> midi = model.generate("song.mp3", composer="composer4")
    >>> midi.write("piano.mid")
"""

from pop2piano.model import Pop2Piano

__version__ = "0.2.0"
__all__ = ["Pop2Piano", "__version__"]
