from typing import Protocol, Optional, runtime_checkable

from .dag import DAG


@runtime_checkable
class CompilerBackend(Protocol):
    def __call__(self, graph: DAG) -> DAG: ...


_registered_backend: Optional[CompilerBackend] = None


def register_backend(fn: CompilerBackend):
    import inspect

    sig = inspect.signature(fn)
    n_params = len(list(sig.parameters.values()))

    if n_params != 1 or not isinstance(fn, CompilerBackend):
        raise TypeError(f"Object {fn} does not implement CompilerBackend protocol")

    global _registered_backend
    _registered_backend = fn
    return fn
