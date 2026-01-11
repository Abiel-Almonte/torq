from .registry import register, register_decorator
from .pipeline import Sequential, Concurrent
from .pipes import Pipe
from .compiler import register_backend, compile

HAS_CUDA = False

try:
    from . import cuda
    HAS_CUDA = True
except ImportError:
    pass