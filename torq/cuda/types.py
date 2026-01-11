from typing import NewType
from types import CapsuleType  # python 3.13

ptr_t = NewType("ptr_t", int)
"""void pointer type"""

handler_t = NewType("handler_t", CapsuleType)
"""CUDA handler type"""

cudaStream_t = NewType("cudaStream_t", handler_t)
"""CUDA Stream handle"""

cudaGraph_t = NewType("cudaGraph_t", handler_t)
"""CUDA Graph handle"""

cudaGraphExec_t = NewType("cudaGraphExec_t", handler_t)
"""CUDA GraphExec handle"""
