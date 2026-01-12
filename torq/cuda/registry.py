from typing import Dict, Callable, Any, Protocol, runtime_checkable

from .types import ptr_t


@runtime_checkable
class ctxFactory(Protocol):
    def __call__(self, ptr: ptr_t) -> Any: ...


_stream_ctx_registry: Dict[str, ctxFactory] = {}


def register_ctx(key: str, ctx_factory: ctxFactory) -> None:
    import inspect

    sig = inspect.signature(ctx_factory)
    params = list(sig.parameters.values())

    if len(params) != 1 or not isinstance(ctx_factory, ctxFactory):
        raise TypeError(f"Object {ctx_factory} does not implement ctxFactory protocol")

    global _stream_ctx_registry
    _stream_ctx_registry[key] = ctx_factory


try:
    import torch

    register_ctx(
        "pytorch", lambda ptr: torch.cuda.stream(torch.cuda.ExternalStream(ptr))
    )  # pytorch preloaded
except ImportError:
    pass

try:
    import cupy as cp

    register_ctx("cupy", lambda ptr: cp.cuda.ExternalStream(ptr))  # cupy preloaded
except ImportError:
    pass
