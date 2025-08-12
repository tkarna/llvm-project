# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import xegpu
from mlir.dialects.transform import structured


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def getDescOp():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        desc_handle = xegpu.GetDescOp(
            sequence.bodyTarget,
            index=0,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: getDescOp
    # CHECK: transform.xegpu.get_desc_op %
    # CHECK: index = 0


@run
def getDescOpDefaultIndex():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        xegpu.GetDescOp(
            sequence.bodyTarget,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: getDescOpDefaultIndex
    # CHECK: transform.xegpu.get_desc_op %


@run
def setResultLayout():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.create_nd_tdesc"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetResultLayoutOp(
            sequence.bodyTarget,
            index=0,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setResultLayout
    # CHECK: %0 = transform.xegpu.set_result_layout %
    # CHECK: index = 0
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


@run
def setResultLayoutDefaultIndex():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.create_nd_tdesc"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetResultLayoutOp(
            sequence.bodyTarget,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setResultLayoutDefaultIndex
    # CHECK: %0 = transform.xegpu.set_result_layout %
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


@run
def setOperandLayout():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("xegpu.dpas"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetOperandLayoutOp(
            sequence.bodyTarget,
            index=0,
            sg_layout=[6, 4],
            sg_data=[32, 16],
            inst_data=[8, 16]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setOperandLayout
    # CHECK: transform.xegpu.set_operand_layout %
    # CHECK: index = 0
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]
    # CHECK: inst_data = [8, 16]


@run
def insertPrefetch():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.AnyOpType.get(),
    )
    with InsertionPoint(sequence.body):
        for_op = structured.MatchOp.match_op_names(sequence.bodyTarget, ["scf.for"])
        dpas_op = structured.MatchOp.match_op_names(for_op, ["xegpu.dpas"])
        xegpu.InsertPrefetchOp(
            dpas_op,
            for_op,
            index=0,
            sg_layout=[6, 4],
            sg_data=[32, 16],
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: insertPrefetch
    # CHECK: %[[FOR_OP:.*]] = transform.structured.match
    # CHECK: %[[DPAS_OP:.*]] = transform.structured.match
    # CHECK: transform.xegpu.insert_prefetch %[[DPAS_OP]] %[[FOR_OP]]
    # CHECK: index = 0
    # CHECK: sg_layout = [6, 4]
    # CHECK: sg_data = [32, 16]


@run
def hoistDescOp():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("scf.for"),
    )
    with InsertionPoint(sequence.body):
        xegpu.HoistDescOp(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: hoistDescOp
    # CHECK: transform.xegpu.hoist_desc_ops


@run
def setGPULaunchThreadsOp():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("gpu.lauch"),
    )
    with InsertionPoint(sequence.body):
        xegpu.SetGPULaunchThreadsOp(
            sequence.bodyTarget,
            threads=[8, 4, 1]
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: setGPULaunchThreadsOp
    # CHECK: transform.xegpu.set_gpu_launch_threads
    # CHECK: threads = [8, 4, 1]
