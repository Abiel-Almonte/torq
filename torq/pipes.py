from types import FunctionType
from typing import Tuple, Callable, Any

import inspect
from .registry import get_registered_types, get_adapter


class Pipe:
    _inner_name = "inner"

    def __init__(self, x: Any) -> None:
        assert isinstance(x, get_registered_types())
        self._inner = x
        self._adapter: Callable[..., Any] = lambda x: x
        self._takes_args: bool = True

    # duck type pipe
    @staticmethod
    def from_opaque(opaque: Any, ins: Any) -> Tuple["Pipe", Any]:
        if not isinstance(opaque, get_registered_types()):
            raise TypeError(f"Pipeline contains an unknown opaque type {type(opaque)}")

        adapter = get_adapter(opaque)
        takes_args = callable(adapter(opaque))

        if takes_args:
            outs = adapter(opaque)(*ins)
        else:
            outs = adapter(opaque)

        has_input = ins is not None and len(ins) != 0
        has_output = outs is not None and len(outs) != 0

        if not isinstance(opaque, FunctionType):
            if not has_input:
                pipe = Input(opaque)
            elif not has_output:
                pipe = Output(opaque)
            elif has_input and has_output:
                pipe = Model(opaque)
            else:
                raise TypeError(
                    f"Unable to reconcile {opaque} into an input, output, model, or functional pipe"
                )
        else:
            pipe = Functional(opaque)

        pipe._adapter = adapter
        pipe._takes_args = takes_args

        return pipe, outs

    def _step(self, *args: Any) -> Any:
        if self._takes_args:
            return self._adapter(self._inner)(*args)
        return self._adapter(self._inner)

    def __call__(self, *args: Any) -> Any:
        return self._step(*args)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inner_name}={self._inner})"


class Input(Pipe):
    _inner_name = "source"


class Model(Pipe):
    _inner_name = "model"


class Output(Pipe):
    _inner_name = "sink"


class Functional(Pipe):
    def __repr__(self) -> str:
        fn_name = getattr(self._inner, "__name__", "function")
        sig = inspect.signature(self._inner)
        return f"Funtional(fn={fn_name}{sig})"
