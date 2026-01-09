from typing import Any, Union, Protocol

from .pipeline import System
from .dag import DAG


class CompilerBackend(Protocol):
    def __call__(self, graph: DAG) -> DAG: ...


def torq_backend(graph: DAG) -> DAG:
    return graph  # dummy


class CompiledSystem:
    def __init__(self, system: System, backend: CompilerBackend) -> None:
        self._system = system
        self._compile = backend
        self._graph: Union[DAG, None] = None

    @property
    def compiled(self) -> bool:
        return self._graph is not None

    def __call__(self, *args: Any) -> Any:
        if not self.compiled:
            ret = self._system(*args)  # run even if already materialized
            self._graph = self._compile(graph=DAG.from_system(self._system))
            return ret

        return self._graph(*args)  # type: ignore

    def run(self, iters: int = -1, *args):
        if not self.compiled:
            self(*args)  # compile, redundant but okay

        return self._graph.run(iters, *args)  # type: ignore


def compile(
    system: System, backend: CompilerBackend = torq_backend
) -> CompiledSystem:  # lazy
    return CompiledSystem(system=system, backend=backend)
