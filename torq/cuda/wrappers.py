from . import _torq as cuda
from .types import cudaStream_t


class CUDAStream:
    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id
        self._handle = cuda.create_stream()

    @property
    def ptr(self):
        return cuda.get_stream_ptr(self._handle)

    def sync(self):
        cuda.sync_stream(self._handle)

    def pytorch_ctx(self):
        try:
            import torch

            return torch.cuda.stream(torch.cuda.ExternalStream(self.ptr))
        except ImportError:
            return None

    def cupy_ctx(self):
        try:
            import cupy as cp

            return cp.cuda.ExternalStream(self.ptr)
        except ImportError:
            return None

    def capture(self, framework="pytorch"):
        """Returns a context manager for graph capture"""
        ctx = None
        if framework == "pytorch":
            ctx = self.pytorch_ctx()
        elif framework == "cupy":
            ctx = self.cupy_ctx()

        return CUDAGraphLauncher(self._handle, ctx)

    def __repr__(self):
        return f"CUDAStream(id={self.stream_id}, ptr={self.ptr:#x})"


class CUDAGraphLauncher:
    def __init__(self, stream_handle: cudaStream_t, stream_ctx=None):
        self._stream_handle = stream_handle
        self._stream_ctx = stream_ctx
        self._graph_handle = None
        self._exec_handle = None

    @property
    def exec_ptr(self):
        return cuda.get_executor_ptr(self._exec_handle) if self._exec_handle else None

    @property
    def graph_ptr(self):
        return cuda.get_graph_ptr(self._graph_handle) if self._graph_handle else None

    def _launch_no_op(self):
        raise RuntimeError("Cannot launch uncaptured CUDA Graph")

    def _launch(self):
        cuda.launch_graph(self._exec_handle, self._stream_handle)

    def launch(self):
        self._launch_no_op()

    def sync(self):
        cuda.sync_stream(self._stream_handle)

    def __enter__(self):
        cuda.begin_capture(self._stream_handle)
        if self._stream_ctx:
            self._stream_ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._stream_ctx:
            self._stream_ctx.__exit__(*args)

        self._graph_handle = cuda.end_capture(self._stream_handle)
        self._exec_handle = cuda.create_executor(self._graph_handle)
        cuda.sync_stream(self._stream_handle)

        self.launch = self._launch

    def __repr__(self):
        return f"CUDAGraphExecutor(status={"captured" if self._exec_handle else "uncaptured"})"
