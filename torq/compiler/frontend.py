from typing import Union, Tuple, List
from collections import defaultdict

from ..core import System, Pipeline, Sequential, Concurrent, Pipe
from ..utils import logging

from .nodes import DAGNode


class _AtomicCounter:
    def __init__(self) -> None:
        self._count = -1

    def get_next(self):
        self._count += 1
        return self._count


class _NameBuilder:
    def __init__(
        self,
        stack: Tuple[str, ...],
        global_cnt: defaultdict,
        local_cnt: defaultdict,
    ) -> None:
        self.stack = stack
        self._global = global_cnt
        self._local = local_cnt

    @property
    def value(self):
        return ".".join(self.stack)

    def _update(self, obj: object, scope: str) -> "_NameBuilder":
        cls_name = obj.__class__.__name__.lower()
        cnt = getattr(self, f"_{scope}")

        idx = cnt[cls_name]
        cnt[cls_name] += 1

        new_stack = self.stack + (f"{cls_name}{idx}",)

        return _NameBuilder(
            stack=new_stack, global_cnt=self._global, local_cnt=self._local.copy()
        )

    def update_global(self, obj: object) -> "_NameBuilder":
        ret = self._update(obj, "global")
        ret._local.clear()
        return ret

    def update_local(self, obj: object) -> "_NameBuilder":
        return self._update(obj, "local")


def lower(system: System):
    nodes = []
    branch_cnt = (
        _AtomicCounter()
    )  # resources are to be assigned to branches, not nodes.

    def _walk(
        pipe: Union[Pipeline, Pipe],
        prev: Union[Tuple[DAGNode, ...], DAGNode, None] = None,
        name: _NameBuilder = None,
        branch: int = 0,
    ) -> Union[DAGNode, Tuple[DAGNode, ...]]:

        if isinstance(pipe, Pipeline):
            pipeline = pipe
            name = name.update_global(pipeline)

            if isinstance(pipeline, Sequential):
                curr = prev
                for pipe in pipeline:
                    curr = _walk(pipe, curr, name, branch)

                return curr

            elif isinstance(pipeline, Concurrent):
                outs = tuple()

                for pipe in pipeline:
                    local_branch = (
                        branch
                        if isinstance(pipe, Concurrent)
                        else branch_cnt.get_next()
                    )
                    out = _walk(pipe, prev, name, local_branch)

                    outs += out if isinstance(out, tuple) else (out,)

                return outs

            else:
                raise TypeError(f"Unknown pipeline type in system: {type(pipeline)}")

        elif isinstance(pipe, Pipe):
            return _lower_pipe(pipe, prev, branch, name, nodes)

        else:
            raise TypeError(f"Unknown type in system: {type(pipe)}")

    leaves = _walk(
        system, name=_NameBuilder(tuple(), defaultdict(int), defaultdict(int))
    )

    if not isinstance(leaves, tuple):
        leaves = (leaves,)  # pack single leaf

    logging.info(f"Graph built with {len(nodes)} nodes and {len(leaves)} trees")

    return nodes, leaves


def _lower_pipe(
    pipe: Pipe, prev, branch: int, name: _NameBuilder, nodes: List[DAGNode]
):
    name = name.update_local(pipe)
    args = tuple()

    if prev:
        if not isinstance(prev, tuple):
            prev = (prev,)

        args += prev

    node = DAGNode(name.value, branch, pipe, args)
    nodes.append(node)

    return node
