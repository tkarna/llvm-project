import atexit

from ..._mlir_libs import get_dialect_registry
from ..._mlir_libs._mlirDialectsTransform._ffi import (
    register_callback_handler,
)

from ...ir import ArrayAttr, Context, SymbolRefAttr, Attribute, Type
from ...dialects import transform
from .._transform_ffi_extension_ops_gen import *

from collections.abc import Sequence
from typing import Union, Dict, Callable, Optional


def callback(
    results: Type,
    name: Union[str, Attribute],
    *payloads: Union[
        transform.AnyOpType, transform.AnyParamType, transform.AnyValueType
    ],
    loc=None,
    ip=None
):
    if isinstance(name, str):
        name = SymbolRefAttr.get([name])

    return CallbackOp(
        results_=results,
        name=name,
        payloads=payloads,
        loc=loc,
        ip=ip,
    )


# Global mapping callback names to Python-function callback functions.
HANDLER_MAPPING: Dict[str, Callable] = {}


# The python function that actually gets called from C++ to deal with
# transform.ffi.callback callbacks.
def callback_handler(name, *args):
    if (handler := HANDLER_MAPPING.get(name)) is None:
        raise RuntimeError(f"callback '{name}' requested but was not registered")
    return handler(*args)


register_callback_handler(callback_handler)
atexit.register(register_callback_handler, None)


# Decorator to register named Python callback functions. Return types need to be
# provided as part of the signature.
def callback_(function: Callable, context: Optional[Context] = None):
    setattr(Context.current, "_callback_handler", callback_handler)

    if function.__name__ in HANDLER_MAPPING:
        raise RuntimeError("tried to register a callback with the same name twice")
    HANDLER_MAPPING[function.__name__] = function
    results_type = function.__annotations__.get("return", ())

    def wrapper(
        *args: Union[
            transform.AnyOpType, transform.AnyValueType, transform.AnyParamType
        ]
    ):
        return callback(results_type, function.__name__, *args)

    return wrapper


# Decorator to register named Python callback function and immediately call it.
# Return types need to be provided as part of the signature.
def call_with(
    *args: Union[transform.AnyOpType, transform.AnyValueType, transform.AnyParamType]
):
    def decorator(function: Callable):
        if function.__name__ in HANDLER_MAPPING:
            raise RuntimeError("tried to register a callback with the same name twice")
        HANDLER_MAPPING[function.__name__] = function
        results_type = function.__annotations__.get("return", ())
        return callback(results_type, function.__name__, *args)

    return decorator
