from .registry import register, register_decorator
from .pipeline import Sequential, Concurrent
from .pipes import Pipe
from .compiler import register_backend, compile
from .utils import logging

try:
    import _torq
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    logging.warning("CUDA extension is not available. Falling back to CPU-only features.")
