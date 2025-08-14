//===- XeGPUTransformOps.td - XeGPU transformation ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XEGPU_EXTENSION
#define XEGPU_EXTENSION

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// This is roughly similar to OpFoldResult assuming the handle produces a single
// value in the payload IR.
def TransformAnyParamTypeOrAnyHandle : Type<
    Or<[TransformHandleTypeInterface.predicate,
        TransformParamTypeInterface.predicate]>,
    "transform any param type or any handle type">;


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
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Set xegpu.layout attribute to an xegpu op result.";
  let description = [{
    Given an xegpu operation, this transform adds `xegpu.layout`
    attribute to it's result's tensor descriptor. The target result is
    defined by the `index` argument. The layout is defined by the
    `sg_layout`, `sg_data` and `inst_data` attributes. If `index` is not
    defined, `index=0` is used. Returns a handle to a transformed op.
  }];

  let arguments = (ins
                   TransformHandleTypeInterface : $target,
                   I64Attr : $resultIndex,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_inst_data
                   );

  let results = (outs TransformHandleTypeInterface : $transformed);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "int64_t":$resultIndex,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData,
                   "ArrayRef<OpFoldResult>":$mixedInstData
                   )>,
  ];

  let assemblyFormat = [{
    $target `index` `=` $resultIndex
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    `inst_data` `=` custom<DynamicIndexList>($inst_data, $static_inst_data)
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticSgLayout(), getSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticSgData(), getSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticInstData(), getInstData(), b);
    }
  }];
}

def SetOpLayoutAttrOp : Op<Transform_Dialect, "xegpu.set_op_layout_attr", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
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
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_inst_data,
                   DefaultValuedAttr<UnitAttr, "false">:$result,
                   DefaultValuedAttr<UnitAttr, "false">:$operand
                   );

  let results = (outs);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "int64_t":$index,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData,
                   "ArrayRef<OpFoldResult>":$mixedInstData,
                   CArg<"bool", "false">:$result,
                   CArg<"bool", "false">:$operand
                   )>,
  ];

  let assemblyFormat = [{
    $target (`result` $result^)? (`operand` $operand^)? (`index` `=` $index^)?
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    `inst_data` `=` custom<DynamicIndexList>($inst_data, $static_inst_data)
    attr-dict `:` type($target) (`,` type($sg_layout)^)? (`,` type($sg_data)^)? (`,` type($inst_data)^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticSgLayout(), getSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticSgData(), getSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticInstData(), getInstData(), b);
    }
  }];
}

def ConvertOperandLayoutOp : Op<Transform_Dialect, "xegpu.convert_operand_layout", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
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
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_inst_data
                   );

  let results = (outs);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "int64_t":$index,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData,
                   "ArrayRef<OpFoldResult>":$mixedInstData
                   )>,
  ];

  let assemblyFormat = [{
    $target `index` `=` $operandIndex
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    `inst_data` `=` custom<DynamicIndexList>($inst_data, $static_inst_data)
    attr-dict `:` type($target) (`,` type($sg_layout)^)? (`,` type($sg_data)^)? (`,` type($inst_data)^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticSgLayout(), getSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticSgData(), getSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticInstData(), getInstData(), b);
    }
  }];
}

def InsertPrefetchOp : Op<Transform_Dialect, "xegpu.insert_prefetch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Adds xegpu prefetch ops to matmul operand tiles.";
  let description = [{
    Given an xegpu operation residing in a `scf.for` loop, this transform inserts cooperative `xegpu.prefetch` operations for the A (index = 0) or B (index = 1) operand. The prefetch tile size is determined by the `sg_layout` and `sg_data` attributes.
  }];

  let arguments = (ins TransformHandleTypeInterface : $target,
                   TransformHandleTypeInterface : $loop,
                   I64Attr : $operandIndex,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle> : $sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data
                   );

  let results = (outs TransformHandleTypeInterface : $transformedTarget,
                 TransformHandleTypeInterface : $transformedLoop);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "Value":$loop,
                   "int64_t":$operandIndex,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData
                   )>,
  ];

  let assemblyFormat = [{
    $target $loop `index` `=` $operandIndex
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticSgLayout(), getSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticSgData(), getSgData(), b);
    }
  }];
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
