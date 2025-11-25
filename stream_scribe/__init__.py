"""Stream Scribe - Real-time speech transcription and conversation structuring."""

try:
    from stream_scribe._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
