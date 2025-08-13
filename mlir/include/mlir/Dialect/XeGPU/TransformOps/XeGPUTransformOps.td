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

def GetDescOp : Op<Transform_Dialect, "xegpu.get_desc_op", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = "Get a handle to the descriptor op of an operand value.";
  let description = [{
    Given an xegpu operation and its operand `index`, returns a handle to the `xegpu.create_nd_tdesc` descriptor op. If `index` is not defined, `index=0` is used.
  }];

  let arguments = (ins TransformHandleTypeInterface : $target,
                   DefaultValuedOptionalAttr<I64Attr, "0"> : $operandIndex);

  let results = (outs TransformHandleTypeInterface : $descHandle);
  let assemblyFormat =
      "$target (`index` `=` $operandIndex^)? attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList & results,
        ::mlir::transform::TransformState & state);
  }];
}

def SetResultLayoutOp : Op<Transform_Dialect, "xegpu.set_result_layout", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = "Set xegpu.layout attribute to an xegpu op result.";
  let description = [{
    Given an xegpu operation, this transform adds `xegpu.layout`
    attribute to it's result's tensor descriptor. The target result is
    defined by the `index` argument. The layout is defined by the
    `sg_layout`, `sg_data` and `inst_data` attributes. If `index` is not
    defined, `index=0` is used. Returns a handle to a transformed op.
  }];

  let arguments = (ins TransformHandleTypeInterface : $target,
                   DefaultValuedOptionalAttr<I64Attr, "0"> : $resultIndex,
                   DenseI32ArrayAttr : $sgLayout,
                   DenseI32ArrayAttr : $sgData,
                   DenseI32ArrayAttr : $instData);

  let results = (outs TransformHandleTypeInterface : $transformed);

  let assemblyFormat =
      "$target (`index` `=` $resultIndex^)? `sg_layout` `=` $sgLayout `sg_data` `=` "
      "$sgData `inst_data` `=` $instData attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList & results,
        ::mlir::transform::TransformState & state);
  }];
}

def SetOpLayoutAttrOp : Op<Transform_Dialect, "xegpu.set_op_layout_attr", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = "Set xegpu.layout attribute of an op.";
  let description = [{
    Sets the `xegpu.layout` attribute of an op. Sets either the
    `layout_result_{index}` or `layout_operand_{index}` attribute. The target
    operand/result value is defined by the `index` argument. The layout is
    defined by the `sg_layout`, `sg_data` and `inst_data` attributes.
  }];

  let arguments = (ins TransformHandleTypeInterface : $target,
                   DefaultValuedOptionalAttr<I64Attr, "0"> : $index,
                   DenseI32ArrayAttr : $sgLayout,
                   DenseI32ArrayAttr : $sgData,
                   DenseI32ArrayAttr : $instData,
                   DefaultValuedAttr<UnitAttr, "false">:$result,
                   DefaultValuedAttr<UnitAttr, "false">:$operand
                   );

  let results = (outs);

  let assemblyFormat =
      "$target (`result` $result^)? (`operand` $operand^)? (`index` `=` $index^)? `sg_layout` `=` $sgLayout `sg_data` `=` "
      "$sgData `inst_data` `=` $instData attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList & results,
        ::mlir::transform::TransformState & state);
  }];
}

def ConvertOperandLayoutOp : Op<Transform_Dialect, "xegpu.convert_operand_layout", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = "Convert xegpu.layout attribute for an xegpu op operand.";
  let description = [{
    Adds an `xegpu.convert_layout` op
    to convert the `xegpu.layout` attribute of an operand. The target operand is
    defined by the `operandIndex` argument. The layout is defined by the
    `sg_layout`, `sg_data` and `inst_data` attributes.
  }];

  let arguments = (ins TransformHandleTypeInterface : $target,
                   I64Attr : $operandIndex,
                   DenseI32ArrayAttr : $sgLayout,
                   DenseI32ArrayAttr : $sgData,
                   DenseI32ArrayAttr : $instData);

  let results = (outs);

  let assemblyFormat =
      "$target `index` `=` $operandIndex `sg_layout` `=` $sgLayout `sg_data` `=` "
      "$sgData `inst_data` `=` $instData attr-dict `:` type($target)";

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
    Given an xegpu operation residing in a `scf.for` loop, this transform inserts cooperative `xegpu.prefetch` operations for the A (index = 0) or B (index = 1) operand. The prefetch tile size is determined by the `sg_layout` and `sg_data` attributes.
  }];

  let arguments = (ins TransformHandleTypeInterface : $target,
                   TransformHandleTypeInterface : $loopOp,
                   I64Attr : $operandIndex,
                   DenseI32ArrayAttr : $sgLayout,
                   DenseI32ArrayAttr : $sgData);

  let results = (outs TransformHandleTypeInterface : $transformedTargetOp,
                 TransformHandleTypeInterface : $transformedLoopOp);

  let assemblyFormat =
      "$target $loopOp `index` `=` $operandIndex `sg_layout` `=` $sgLayout `sg_data` `=` "
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

  let arguments = (ins TransformHandleTypeInterface : $launchOp,
                   DenseI32ArrayAttr : $threads);
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
