//===- XeGPUTransformOps.td - XeGPU transformation ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XEGPU_EXTENSION
#define XEGPU_EXTENSION

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HoistDescOp : Op<Transform_Dialect, "xegpu.hoist_desc_ops", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = "Hoists xegpu tile descriptor ops outside the containing loop";
  let description = [{
    Hoists `xepu.create_nd_tdesc` out of the loop. If the
    descriptor's offset is loop dependent, a `xegpu.update_nd_offset` op is
    inserted in the loop to increment the offset.
  }];

  let arguments = (ins TransformHandleTypeInterface : $loop);
  let results = (outs TransformHandleTypeInterface : $transformed);

  let assemblyFormat = "$loop attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList & results,
        ::mlir::transform::TransformState & state);
  }];
}

def SetDPASLayoutOp : Op<Transform_Dialect, "xegpu.set_dpas_layout", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = "Set xegpu.layout attribute to an DPAS op operand.";
  let description = [{
    Given a `xegpu.dpas` operation, this transform adds `xegpu.layout`
    attribute to it's operand's tensor descriptor. The target operand is
    defined by the `tileIndex` argument. The layout is defined by the
    `sg_layout`, `sg_data` and `inst_data` attributes. The `load_data`
    attribute defines the tile size used for loading the data. It must be a
    multiple of the `inst_data` size.
  }];

  let arguments = (ins TransformHandleTypeInterface : $dpasOp,
                   I64Attr : $tileIndex,
                   DenseI32ArrayAttr : $sgLayout,
                   DenseI32ArrayAttr : $sgData,
                   OptionalAttr<DenseI32ArrayAttr> : $loadData,
                   DenseI32ArrayAttr : $instData);

  let results = (outs);

  let assemblyFormat =
      "$dpasOp `index` `=` $tileIndex `sg_layout` `=` $sgLayout `sg_data` `=` "
      "$sgData (`load_data` `=` $loadData^)? `inst_data` `=` $instData attr-dict `:` type($dpasOp)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList & results,
        ::mlir::transform::TransformState & state);
  }];
}

def InsertPrefetchOp : Op<Transform_Dialect, "xegpu.insert_prefetch",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>]> {

  let summary = "Adds xegpu prefetch ops to matmul operand tiles.";
  let description = [{
    Given a `xegpu.dpas` operation residing in a `scf.for` loop, this transform inserts cooperative `xegpu.prefetch` operations for the A (index = 0) or B (index = 1) operand. The prefetch tile size is determined by the `sg_layout` and `sg_data` attributes.
  }];

  let arguments = (ins TransformHandleTypeInterface : $dpasOp,
                   TransformHandleTypeInterface : $loopOp,
                   I64Attr : $tileIndex,
                   DenseI32ArrayAttr : $sgLayout,
                   DenseI32ArrayAttr : $sgData);

  let results = (outs TransformHandleTypeInterface : $transformedDpasOp,
                 TransformHandleTypeInterface : $transformedLoopOp);

  let assemblyFormat =
      "$dpasOp $loopOp `index` `=` $tileIndex `sg_layout` `=` $sgLayout `sg_data` `=` "
      "$sgData attr-dict `:` functional-type(operands, results)";
}

// TODO this should be handled with gpu transform ops.
// Add gpu mapping to scf.forall op and use something like
// transform.gpu.map_forall_to_blocks to convert to gpu.launch op.
def SetGPULaunchThreadsOp
    : Op<Transform_Dialect, "xegpu.set_gpu_launch_threads", [
      TransformOpInterface, TransformEachOpTrait,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    ]> {

  let summary = "Set number of threads for a given gpu.launch operation";
  let description = [{Set number of threads for a given gpu.launch operation}];

  let arguments = (ins TransformHandleTypeInterface
                   : $launchOp, DenseI32ArrayAttr
                   : $threads);
  let results = (outs);
  let assemblyFormat =
      "$launchOp `threads` `=` $threads attr-dict `:` type($launchOp)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList & results,
        ::mlir::transform::TransformState & state);
  }];
}

#endif // XEGPU_EXTENSION
