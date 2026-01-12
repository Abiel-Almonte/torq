from typing import Optional

from ..utils import logging

from .handlers import CUDAStream
from .launcher import GraphLauncher
from .registry import _stream_ctx_registry


class Stream:
    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id
        self._stream = CUDAStream()

    def synchronize(self) -> None:
        self._stream.synchronize()

    def capture(self, framework: Optional[str] = "pytorch") -> Optional[GraphLauncher]:
        if framework is None:  # exhaustive search

            for ctx_factory in _stream_ctx_registry.values():
                try:
                    ctx = ctx_factory(self._stream.ptr)
                    return GraphLauncher(self._stream, ctx)
                except Exception:
                    continue

            logging.warning("No framework context worked, using raw CUDA capture")
            return GraphLauncher(self._stream, None)

        else:
            if framework not in _stream_ctx_registry:
                raise ValueError(
                    f"Framework {framework} not registered",
                    f"Available frameworks {list(_stream_ctx_registry.keys())}",
                    "Register your framework using `tq.cuda.register`",
                )

            ctx_factory = _stream_ctx_registry[framework]
            ctx = ctx_factory(self._stream.ptr)
            return GraphLauncher(self._stream, ctx)

    def __repr__(self) -> str:
        return f"Stream(id={self.stream_id}, stream={self._stream})"
