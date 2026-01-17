from typing import Tuple, Union


class Node:
    id: str
    branch: int


Nodes = Tuple[Node, ...]
NodeOrNodes = Union[Node, Nodes]
