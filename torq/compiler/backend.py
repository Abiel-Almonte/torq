from .registry import register_backend
from .dag import DAG


@register_backend
def torq_backend(graph: DAG) -> DAG:
    return graph  # TODO
