#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._xegpu_transform_ops_gen import *
from .._xegpu_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import _cext as _ods_cext
    from .._ods_common import (
        MixedValues,
        get_op_result_or_value as _get_op_result_or_value,
        _dispatch_dynamic_index_list,
    )

except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union, overload


@_ods_cext.register_operation(_Dialect, replace=True)
class GetDescOp(GetDescOp):
    """Specialization for GetDescOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        *,
        index: Optional[Union[int, Attribute]] = None,
        loc=None,
        ip=None,
    ):
        desc_type = transform.AnyOpType.get()
        super().__init__(
            desc_type,
            _get_op_result_or_value(target),
            operandIndex=index,
            loc=loc,
            ip=ip
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class SetResultLayoutOp(SetResultLayoutOp):
    """Specialization for SetResultLayoutOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        index: Union[int, Attribute],
        sg_layout: MixedValues,
        sg_data: MixedValues,
        inst_data: MixedValues,
        *,
        loc=None,
        ip=None,
    ):
        target_value = _get_op_result_or_value(target)
        (
            dynamic_sg_layout,
            static_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(sg_layout)
        (
            dynamic_sg_data,
            static_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(sg_data)
        (
            dynamic_inst_data,
            static_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(inst_data)

        super().__init__(
            target_value.type,
            target_value,
            index,
            dynamic_sg_layout,
            dynamic_sg_data,
            dynamic_inst_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            static_inst_data=static_inst_data,
            loc=loc,
            ip=ip
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class SetOpLayoutAttrOp(SetOpLayoutAttrOp):
    """Specialization for SetOpLayoutAttrOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        sg_layout: MixedValues,
        sg_data: MixedValues,
        inst_data: MixedValues,
        *,
        index: Union[int, Attribute] = None,
        result: Union[bool, Attribute] = None,
        operand: Union[bool, Attribute] = None,
        loc=None,
        ip=None,
    ):
        (
            dynamic_sg_layout,
            static_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(sg_layout)
        (
            dynamic_sg_data,
            static_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(sg_data)
        (
            dynamic_inst_data,
            static_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(inst_data)
        if result is None and operand is None:
            result = True
        super().__init__(
            _get_op_result_or_value(target),
            dynamic_sg_layout,
            dynamic_sg_data,
            dynamic_inst_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            static_inst_data=static_inst_data,
            index=index,
            result=result,
            operand=operand,
            loc=loc,
            ip=ip
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class ConvertOperandLayoutOp(ConvertOperandLayoutOp):
    """Specialization for ConvertOperandLayoutOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        index: Union[int, Attribute],
        sg_layout: MixedValues,
        sg_data: MixedValues,
        inst_data: MixedValues,
        *,
        loc=None,
        ip=None,
    ):
        (
            dynamic_sg_layout,
            static_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(sg_layout)
        (
            dynamic_sg_data,
            static_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(sg_data)
        (
            dynamic_inst_data,
            static_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(inst_data)
        super().__init__(
            target,
            index,
            dynamic_sg_layout,
            dynamic_sg_data,
            dynamic_inst_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            static_inst_data=static_inst_data,
            loc=loc,
            ip=ip
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class InsertPrefetchOp(InsertPrefetchOp):
    """Specialization for InsertPrefetchOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        loop_op: Union[Operation, Value],
        index: Union[int, Attribute],
        sg_layout: MixedValues,
        sg_data: MixedValues,
        loc=None,
        ip=None,
    ):
        (
            dynamic_sg_layout,
            static_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(sg_layout)
        (
            dynamic_sg_data,
            static_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(sg_data)
        transformed_target_type = transform.AnyOpType.get()
        transformed_loop_type = transform.AnyOpType.get()
        super().__init__(
            transformed_target_type,
            transformed_loop_type,
            _get_op_result_or_value(target),
            _get_op_result_or_value(loop_op),
            index,
            dynamic_sg_layout,
            dynamic_sg_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            loc=loc,
            ip=ip
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class HoistDescOp(HoistDescOp):
    """Specialization for HoistDescOp class."""

    def __init__(
        self,
        loop_op: Union[Operation, Value],
        loc=None,
        ip=None,
    ):
        transformed_loop_type = transform.AnyOpType.get()
        super().__init__(
            transformed_loop_type,
            _get_op_result_or_value(loop_op),
            loc=loc,
            ip=ip
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class SetGPULaunchThreadsOp(SetGPULaunchThreadsOp):
    """Specialization for SetGPULaunchThreadsOp class."""

    def __init__(
        self,
        launch_op: Union[Operation, Value],
        threads: Union[int, Attribute],
        loc=None,
        ip=None,
    ):
        super().__init__(
            _get_op_result_or_value(launch_op),
            threads,
            loc=loc,
            ip=ip
        )
