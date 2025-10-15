"""
Matmul example using linalg->xegpu->xevm lowering.

MLIR must be compiled with level zero, spirv and python bindings, i.e.
something like:

cmake -G Ninja ../llvm \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=<llvm_install_path> \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV -DLLVM_INSTALL_GTEST=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_ENABLE_BINDINGS=OFF \
   -DLLVM_USE_LINKER=lld \
   -DLLVM_ENABLE_RTTI=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=1 \
   -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
   -DPython3_EXECUTABLE=$(which python3)

Python bindings require: nanobind pybind11
Python environment (e.g. virtualenv) must have: PyYAML

Run environment:
LLVM_ROOT=<llvm_install_path>
export PATH=$LLVM_ROOT/bin:$PATH
export PYTHONPATH=$LLVM_ROOT/python_packages/mlir_core

Expected performance on BMG B580:
4096,4096,4096 f16,f16,f32 wg_tile=256,256 sg_tile=32,32 k_tile=32 load_tile_a=32,16 load_tile_b=32,16 prefetch_tile_a=8,32 prefetch_tile_b=8,32 nb_prefetch=1 time (ms): 1.509 GFLOPS/s: 90576.49

Tested with LLVM based on 1098a5cefd764eb58e8530e821eaa5d5a6c42310
"""
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop
from mlir.dialects.transform import interpreter as transform_interpreter
from mlir.dialects.transform import structured
from mlir.dialects.transform import loop
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import xegpu
from mlir.dialects.bufferization import LayoutMapOption
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
import subprocess
import numpy as np
import ctypes
import os
from functools import cached_property
from typing import Union
import argparse


def get_mlir_install_path():
    pkg_path = ir.__file__
    key = "python_packages"
    assert key in pkg_path
    return pkg_path.split(key)[0]


def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    return structured.MatchOp(transform.AnyOpType.get(), *args, **kwargs)


def cse(op):
    transform.ApplyCommonSubexpressionEliminationOp(op)


def canonicalize(op):
    with ir.InsertionPoint(transform.ApplyPatternsOp(op).patterns):
        transform.ApplyCanonicalizationPatternsOp()


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Matrix Multiplication using MLIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[4096, 4096, 4096],
        help="M,N,K matrix sizes (A=MxK, B=KxN, C=MxN).",
    )
    parser.add_argument(
        "--wg-tile",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Workgroup tile size M,N.",
    )
    parser.add_argument(
        "--sg-tile",
        type=int,
        nargs=2,
        default=[32, 32],
        help="Subgroup tile size M,N.",
    )
    parser.add_argument(
        "--k-tile",
        type=int,
        default=32,
        help="Inner reduction dimension tile size K.",
    )
    parser.add_argument(
        "--load-tile-a",
        type=int,
        nargs=2,
        default=[32, 16],
        help="Tile size for loading A matrix for DPAS op.",
    )
    parser.add_argument(
        "--load-tile-b",
        type=int,
        nargs=2,
        default=[32, 16],
        help="Tile size for loading B matrix for DPAS op.",
    )
    parser.add_argument(
        "--prefetch-tile-a",
        type=int,
        nargs=2,
        default=[8, 32],
        help="Tile size for cooperative prefetching of subgroup A matrix",
    )
    parser.add_argument(
        "--prefetch-tile-b",
        type=int,
        nargs=2,
        default=[8, 16],
        help="Tile size for cooperative prefetching of subgroup B matrix",
    )
    parser.add_argument(
        "--nb-prefetch",
        type=int,
        default=1,
        help="Number of initial prefetches.",
    )
    parser.add_argument(
        "--ab-type",
        type=str,
        choices=["f16", "f32"],
        default="f16",
        help="Data type of A and B matrices.",
    )
    parser.add_argument(
        "--c-type",
        type=str,
        choices=["f16", "f32"],
        default="f32",
        help="Data type of the C matrix.",
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Add relu op after the matrix multiplication.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "xegpu-wg",
            "xegpu-sg",
            "xegpu-inst",
            "final",
        ],
        help="Dump kernel IR at different stages of lowering.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    parser.add_argument(
        "--dump-timing",
        action="store_true",
        help="Print timing information for different stages.",
    )
    args = parser.parse_args()

    return args


# FIXME generate payload IR using python
payload_matmul = """
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %3 = linalg.matmul ins(%0, %1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
"""

payload_matmul_relu = """
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %3 = linalg.matmul ins(%0, %1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %4 = tensor.empty() : tensor<4096x4096xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %6 = linalg.max ins(%3, %5 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    bufferization.materialize_in_destination %6 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
"""


class MatMul:
    def __init__(self, M: int = 4096, N: int = 4096, K: int = 4096,
                 ab_type: str = "f16", c_type: str = "f32",
                 has_relu: bool = False,
                 dump_kernel: Union[str, None] = None,
                 dump_schedule: Union[bool, None] = None,
                 tune_params: Union[dict, None] = None):
        self.M = M
        self.N = N
        self.K = K
        self.ab_type = ab_type
        self.c_type = c_type
        self.has_relu = has_relu
        self.dump_kernel = dump_kernel
        self.dump_schedule = dump_schedule
        assert (M, N, K) == (4096, 4096, 4096), "only 4096x4096x4096 size is supported"
        assert (ab_type, c_type) == ("f16", "f32"), "only f16,f16,f32 dtypes are supported"
        self.tune_params = {
            "wg_tile": [256, 256],
            "sg_tile": [32, 64],
            "k_tile": 32,
            "load_tile_a": [8, 16],
            "load_tile_b": [16, 16],
            "prefetch_tile_a": [8, 32],
            "prefetch_tile_b": [8, 32],
            "nb_prefetch": 1
        }
        if tune_params is not None:
            self.tune_params.update(tune_params)
        self.ctx = ir.Context()
        self.loc = ir.Location.unknown(context=self.ctx)

    @cached_property
    def payload_module(self):
        with self.ctx, self.loc:
            # payload module
            payload_file = "payload_main.mlir"
            with open(payload_file, "r") as f:
                payload_ir = f.read()
            # insert payload func
            func_ir = payload_matmul_relu if self.has_relu else payload_matmul
            payload_ir = payload_ir.replace("// ##PAYLOAD_FUNC##", func_ir)
            mod = ir.Module.parse(payload_ir)
        return mod

    @cached_property
    def schedule_module(self):
        with self.ctx, self.loc:
            mod = ir.Module.create()
            mod.operation.attributes["transform.with_named_sequence"] = (
                ir.UnitAttr.get()
            )
            with ir.InsertionPoint(mod.body):
                named_sequence = transform.NamedSequenceOp(
                    "__transform_main",
                    [transform.AnyOpType.get()],  # input types
                    [],  # output types
                    arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
                )
                with ir.InsertionPoint(named_sequence.body):
                    self.emit_transform_schedule(named_sequence)
        return mod

    @cached_property
    def module(self):
        with self.ctx, self.loc:
            mod = ir.Module.create()
            mod.body.append(self.payload_module.operation)
            mod.body.append(self.schedule_module.operation)
        return mod

    def emit_transform_schedule(self, named_sequence):
        anytype = transform.AnyOpType.get()
        func = match(named_sequence.bodyTarget, ops={"func.func"})
        mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )

        # hardware constraints
        dpas_tile = [8, 16, 16]
        prefetch_inst_data = [8, 16]
        nb_workitems = 16  # workitems in subgroup

        # tunable parameters
        wg_tile = self.tune_params["wg_tile"]
        sg_tile = self.tune_params["sg_tile"]
        k_tile = self.tune_params["k_tile"]

        load_tile_a = self.tune_params["load_tile_a"]
        load_tile_b = self.tune_params["load_tile_b"]

        prefetch_tile_a = self.tune_params["prefetch_tile_a"]
        prefetch_tile_b = self.tune_params["prefetch_tile_b"]
        nb_prefetch = self.tune_params["nb_prefetch"]

        # derived parameters
        sg_layout = [wg_tile[0] // sg_tile[0], wg_tile[1] // sg_tile[1]]
        # number of threads collapsed to 1d layout
        nb_threads = sg_layout[0] * sg_layout[1] * nb_workitems
        prefetch_layout_a = [
            wg_tile[0] // prefetch_tile_a[0],
            k_tile // prefetch_tile_a[1]
        ]
        prefetch_layout_b = [
            k_tile // prefetch_tile_b[0],
            wg_tile[1] // prefetch_tile_b[1]
        ]

        # matmul matrix shapes
        sg_tile_a = [sg_tile[0], k_tile]
        sg_tile_b = [k_tile, sg_tile[1]]
        dpas_shape_a = [dpas_tile[0], dpas_tile[2]]
        dpas_shape_b = [dpas_tile[2], dpas_tile[1]]
        dpas_shape_c = [dpas_tile[0], dpas_tile[1]]

        if self.dump_kernel == "initial":
            transform.YieldOp()
            return

        # apply transformations

        if self.has_relu:
            # tile leaf and progressively tile-fuse producers
            # FIXME structured.FuseOp can now fuse everything in one step
            max_op = match(mod, ops={"linalg.max"})
            wg_max, wg_loop = structured.TileUsingForallOp(
                max_op, tile_sizes=wg_tile
            ).results
            fill = match(mod, ops={"linalg.fill"})
            structured.FuseIntoContainingOp(fill, wg_loop)
            matmul = match(mod, ops={"linalg.matmul"})
            func = transform.get_parent_op(anytype, matmul)
            structured.FuseIntoContainingOp(matmul, wg_loop)
            cse(mod)
            canonicalize(mod)
        else:
            matmul = match(mod, ops={"linalg.matmul"})
            func = transform.get_parent_op(anytype, matmul)
            wg_matmul, wg_loop = structured.TileUsingForallOp(
                matmul, tile_sizes=wg_tile
            ).results

        # k loop tiling
        wg_matmul = match(mod, ops={"linalg.matmul"}).result
        wgk_matmul, k_loop = structured.TileUsingForOp(
            wg_matmul, sizes=[0, 0, k_tile]
        ).results

        cse(func)
        canonicalize(func)

        if self.dump_kernel == "tiled":
            transform.YieldOp()
            return

        # vectorize
        func = structured.VectorizeChildrenAndApplyPatternsOp(
            func,
            fold_type_extensions_into_contract=True,
        ).result

        # hoist loop invariant vector read/store ops
        k_loop = match(func, ops={"scf.for"})
        loop.HoistLoopInvariantSubsetsOp(k_loop)

        cse(func)
        canonicalize(func)

        if self.dump_kernel == "vectorized":
            transform.YieldOp()
            return

        # bufferize
        identity_layout = LayoutMapOption.IdentityLayoutMap
        mod = bufferization.OneShotBufferizeOp(
            mod,
            allow_return_allocs_from_loops=True,
            bufferize_function_boundaries=True,
            function_boundary_type_conversion=identity_layout,
        ).result
        # fold memref.subviews into vector.transfer_read/write ops
        mod = apply_registered_pass(mod, "fold-memref-alias-ops")
        cse(mod)
        canonicalize(mod)
        # FIXME pass does not exist upstream
        # mod = apply_registered_pass(mod, "imex-remove-temporaries")
        mod = apply_registered_pass(mod, "drop-equivalent-buffer-results")

        if self.dump_kernel == "bufferized":
            transform.YieldOp()
            return

        # convert forall to parallel
        wg_loop = match(mod, ops={"scf.forall"})
        wg_loop = loop.ForallToParallelOp([anytype], wg_loop)
        func = transform.get_parent_op(anytype, wg_loop)

        # convert to scf.parallel to gpu.launch
        func = apply_registered_pass(func, "gpu-map-parallel-loops")
        func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
        func = apply_registered_pass(func, "lower-affine")
        cse(func)
        canonicalize(func)

        # set correct number of gpu threads
        launch_op = match(func, ops={"gpu.launch"})
        xegpu.SetGPULaunchThreadsOp(launch_op, threads=[nb_threads, 1, 1])

        # outline gpu func
        func = apply_registered_pass(func, "lower-affine")
        canonicalize(func)
        func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
        mod = apply_registered_pass(mod, "gpu-kernel-outlining")
        cse(mod)

        # set xevm target
        mod = apply_registered_pass(
            mod,
            "xevm-attach-target",
            options={"O": "3", "chip": "bmg"},
        )

        # convert vector to xegpu
        gpu_mod = match(mod, ops={"gpu.module"})
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        cse(gpu_func)

        # add layouts to DPAS op operands
        k_loop = match(gpu_func, ops={"scf.for"})
        dpas_op = match(k_loop, ops={"xegpu.dpas"})

        # matmul matrix shapes
        sg_tile_a = [sg_tile[0], k_tile]
        sg_tile_b = [k_tile, sg_tile[1]]
        dpas_shape_a = [dpas_tile[0], dpas_tile[2]]
        dpas_shape_b = [dpas_tile[2], dpas_tile[1]]
        dpas_shape_c = [dpas_tile[0], dpas_tile[1]]

        # A tile load layout
        layout_load_a = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_a,
            "inst_data": load_tile_a,
        }
        desc_op_a = xegpu.GetDescOp(target=dpas_op, index=0)
        desc_op_a = xegpu.SetDescLayoutOp(
            target=desc_op_a,
            index=0,
            **layout_load_a,
        )
        # A tile dpas layout
        layout_dpas_a = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_a,
            "inst_data": dpas_shape_a,
        }
        xegpu.ConvertOperandLayoutOp(target=dpas_op, index=0, **layout_dpas_a)

        # B tile load layout
        layout_load_b = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_b,
            "inst_data": load_tile_b,
        }
        desc_op_b = xegpu.GetDescOp(target=dpas_op, index=1)
        desc_op_b = xegpu.SetDescLayoutOp(
            target=desc_op_b,
            index=0,
            **layout_load_b,
        )
        # B tile dpas layout
        layout_dpas_b = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_b,
            "inst_data": dpas_shape_b,
        }
        xegpu.ConvertOperandLayoutOp(target=dpas_op, index=1, **layout_dpas_b)

        # C tile layout
        output_layout = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile,
            "inst_data": dpas_shape_c,
        }
        desc_op_c = xegpu.GetDescOp(target=dpas_op, index=2)
        desc_op_c = xegpu.SetDescLayoutOp(target=desc_op_c, index=0, **output_layout)
        # C tile dpas layout
        xegpu.SetOpLayoutAttrOp(target=dpas_op, result=True, index=0, **output_layout)

        if self.has_relu:
            # for post ops we need to add C layout manually
            max_op = match(gpu_func, ops={"arith.maximumf"}).result
            xegpu.SetOpLayoutAttrOp(target=max_op, result=True, index=0, **output_layout)
            # find zero constant buffer and annotate it
            const_buffer = transform.get_producer_of_operand(anytype, max_op, 1)
            xegpu.SetOpLayoutAttrOp(
                target=const_buffer, result=True, index=0, **output_layout
            )

        # insert prefetch ops for DPAS A and B tiles
        xegpu.InsertPrefetchOp(
            target=dpas_op,
            loop_op=k_loop,
            index=0,
            nb_prefetch=nb_prefetch,
            sg_layout=prefetch_layout_a,
            sg_data=prefetch_tile_a,
            inst_data=prefetch_inst_data,
        )
        xegpu.InsertPrefetchOp(
            target=dpas_op,
            loop_op=k_loop,
            index=1,
            nb_prefetch=nb_prefetch,
            sg_layout=prefetch_layout_b,
            sg_data=prefetch_tile_b,
            inst_data=prefetch_inst_data,
        )
        cse(gpu_func)
        canonicalize(gpu_func)

        # hoist desc ops out of reduction loop
        transform.apply_licm(k_loop)

        canonicalize(gpu_func)
        cse(gpu_func)

        if self.dump_kernel == "xegpu-wg":
            transform.YieldOp()
            return

        # xegpu distribution
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "xegpu-wg-to-sg-distribute")
        cse(gpu_func)

        if self.dump_kernel == "xegpu-sg":
            transform.YieldOp()
            return

        gpu_func = apply_registered_pass(gpu_func, "lower-affine")
        cse(gpu_func)
        gpu_func = apply_registered_pass(gpu_func, "xegpu-blocking")
        canonicalize(gpu_func)
        cse(gpu_func)

        if self.dump_kernel == "xegpu-inst":
            transform.YieldOp()
            return

        gpu_func = apply_registered_pass(gpu_func, "xegpu-propagate-layout")
        gpu_mod = apply_registered_pass(gpu_mod, "xegpu-subgroup-distribute")
        canonicalize(gpu_mod)
        cse(gpu_mod)
        gpu_mod = apply_registered_pass(gpu_mod, "loop-invariant-code-motion")
        cse(gpu_mod)
        gpu_mod = apply_registered_pass(gpu_mod, "xegpu-vector-linearize")
        gpu_mod = apply_registered_pass(gpu_mod, "convert-xegpu-to-xevm")
        gpu_mod = apply_registered_pass(gpu_mod, "convert-gpu-to-llvm-spv",
                                        options={"use-64bit-index": "true"})
        gpu_mod = apply_registered_pass(gpu_mod, "convert-xevm-to-llvm")
        cse(gpu_mod)

        func = match(mod, ops={"func.func"})
        func = apply_registered_pass(func, "gpu-async-region")

        mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
        mod = apply_registered_pass(mod, "convert-vector-to-scf")
        mod = apply_registered_pass(mod, "convert-scf-to-cf")
        mod = apply_registered_pass(mod, "expand-strided-metadata")
        mod = apply_registered_pass(mod, "finalize-memref-to-llvm")
        mod = apply_registered_pass(mod, "convert-cf-to-llvm")
        mod = apply_registered_pass(mod, "convert-vector-to-llvm")
        mod = apply_registered_pass(mod, "convert-arith-to-llvm")
        mod = apply_registered_pass(mod, "convert-index-to-llvm")
        mod = apply_registered_pass(mod, "convert-func-to-llvm")
        mod = apply_registered_pass(mod, "convert-math-to-llvm")
        mod = apply_registered_pass(mod, "gpu-to-llvm")
        mod = apply_registered_pass(mod, "lower-affine")
        mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
        cse(mod)
        mod = apply_registered_pass(mod, "gpu-module-to-binary")

        transform.YieldOp()

    def compile(self):
        with self.ctx, self.loc:
            # invoke transform interpreter directly
            main_named_sequence = self.schedule_module.body.operations[0]
            transform_interpreter.apply_named_sequence(
                payload_root=self.payload_module,
                transform_root=main_named_sequence,
                transform_module=self.module,
            )
        if self.dump_kernel:
            print(self.payload_module)
        if self.dump_schedule:
            print(self.schedule_module)

    def execute(self):
        if self.dump_kernel or self.dump_schedule:
            return
        libs = [
            "libmlir_c_runner_utils.so",
            "libmlir_runner_utils.so",
            "libmlir_levelzero_runtime.so",
        ]
        mlir_install_path = get_mlir_install_path()
        print(f"MLIR install path: {mlir_install_path}")
        lib_dir = os.path.join(mlir_install_path, "lib")
        libs = [os.path.join(lib_dir, f) for f in libs]

        # run using execution engine -- broken at the moment
        # with self.ctx, self.loc:
        #     execution_engine = ExecutionEngine(
        #         self.payload_module, opt_level=3, shared_libs=libs
        #     )
        #     main_func = execution_engine.lookup("main")
        #     # for some reason function requires one void pointer argument
        #     A = np.ndarray((1, ), dtype=np.float16)
        #     argA = ctypes.pointer(
        #         ctypes.pointer(get_ranked_memref_descriptor(A)))
        #     # call
        #     main_func(argA)

        # run with mlir-runner
        with open("input.mlir", "w") as f:
            f.write(str(self.payload_module))

        cmd = [
            "mlir-runner",
            "-e",
            "main",
            "--entry-point-result=void",
            "--shared-libs=" + ",".join(libs),
            "input.mlir",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Command output:\n{result.stdout}")
            print(f"Command error:\n{result.stderr}")
            raise RuntimeError(f"mlir-runner failed with exit code {result.returncode}")
        output_str = result.stdout

        def parse_time(output_str, key):
            for line in output_str.split("\n"):
                if key in line:
                    time = float(line.split()[-1])
                    return time
            return None

        time = parse_time(output_str, "Average time in kernel: ")
        flops = 2 * self.M * self.N * self.K
        if self.has_relu:
            flops += self.M * self.N
        gflopsps = flops / time / 1e9
        print(f"{self.M},{self.N},{self.K}", end=" ")
        print(f"{self.ab_type},{self.ab_type},{self.c_type}", end=" ")
        print(params_string(self.tune_params), end=" ")
        print(f"time (ms): {time*1000:.3f} GFLOPS/s: {gflopsps:.2f}")


def params_string(tune_params):
    def list2str(a):
        return ",".join(map(str, a))
    out = ""
    for k, v in tune_params.items():
        if isinstance(v, list):
            v = list2str(v)
        out += f"{k}={v} "
    return out.rstrip()


def parse_args_and_run():
    args = parse_cli()
    tune_params = {
        "wg_tile": args.wg_tile,
        "sg_tile": args.sg_tile,
        "k_tile": args.k_tile,
        "load_tile_a": args.load_tile_a,
        "load_tile_b": args.load_tile_b,
        "prefetch_tile_a": args.prefetch_tile_a,
        "prefetch_tile_b": args.prefetch_tile_b,
        "nb_prefetch": args.nb_prefetch,
    }
    kernel = MatMul(
        M=args.sizes[0],
        N=args.sizes[1],
        K=args.sizes[2],
        ab_type=args.ab_type,
        c_type=args.c_type,
        has_relu=args.relu,
        dump_kernel=args.dump_kernel,
        dump_schedule=args.dump_schedule,
        tune_params=tune_params,
    )
    kernel.compile()
    kernel.execute()


if __name__ == "__main__":
    parse_args_and_run()
