from typing import Optional, Union, Tuple
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


def lower(system: System):
    nodes = tuple()
    branch_cnt = (
        _AtomicCounter()
    )  # resources are to be assigned to branches, not nodes.
    global_cnt = defaultdict(int)

    def walk(
        pipe: Union[Pipeline, Pipe],
        prev: Union[Tuple[DAGNode, ...], DAGNode, None] = None,
        name: str = "",
        branch: int = 0,
        local_cnt: Optional[defaultdict] = None,
    ) -> Union[DAGNode, Tuple[DAGNode, ...]]:

        if local_cnt is None:
            local_cnt = defaultdict(int)

        cls_name = str(pipe.__class__.__name__)
        base = f"{name}.{cls_name.lower()}" if name else cls_name.lower()

        if isinstance(pipe, Pipeline):
            pipeline = pipe
            name = base + str(global_cnt[cls_name])

            global_cnt[cls_name] += 1
            local_cnt.clear()
            if isinstance(pipeline, Sequential):
                curr = prev
                for pipe in pipeline:
                    curr = walk(pipe, curr, name, branch, local_cnt)

                if curr is None:
                    raise RuntimeError(f"Invalid pipeline. Sequential is empty")

                return curr

            elif isinstance(pipeline, Concurrent):
                outs = tuple()

                for pipe in pipeline:
                    out = walk(
                        pipe,
                        prev,
                        name,
                        branch=(
                            branch
                            if isinstance(pipe, Concurrent)
                            else branch_cnt.get_next()
                        ),
                        local_cnt=local_cnt,
                    )

                    if not isinstance(out, tuple):
                        out = (out,)

                    outs += out

                if len(outs) == 0:
                    raise RuntimeError(f"Invalid pipeline. Concurrent is empty")

                return outs

            else:
                raise TypeError(f"Unknown pipeline type in system: {type(pipeline)}")

        elif isinstance(pipe, Pipe):
            name = base + str(local_cnt[cls_name])
            local_cnt[cls_name] += 1

            args = tuple()

            if prev:
                if not isinstance(prev, tuple):
                    prev = (prev,)

                args += prev

            node = DAGNode(name, branch, pipe, args)

            nonlocal nodes
            nodes += (node,)

            return node

        else:
            raise TypeError(f"Unknown type in system: {type(pipe)}")

    leaves = walk(system)

    if not isinstance(leaves, tuple):
        leaves = (leaves,)  # pack single leaf

    logging.info(f"Graph built with {len(nodes)} nodes and {len(leaves)} trees")

    return nodes, leaves
