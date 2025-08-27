//===- FFIExtensionOps.cpp - FFI extension for the Transform dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/FFIExtension/FFIExtensionOps.h"
#include "mlir/Dialect/Transform/FFIExtension/FFIExtension.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/FFIExtension/FFIExtensionOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CallbackOp
//===----------------------------------------------------------------------===//

void transform::ffi::CallbackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getPayloadsMutable(), // TODO: Make specifiable on the op.
                  effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

DiagnosedSilenceableFailure
transform::ffi::CallbackOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  Handler handler = transform::ffi::getHandler();
  if (handler == nullptr)
    return emitDefiniteFailure()
           << "callback called without a registered callback handler";

  SmallVector<SmallVector<MappedValue>> payloads;
  transform::detail::prepareValueMappings(payloads, getPayloads(), state);

  SmallVector<SmallVector<MappedValue>> res =
      handler(getName().getRootReference().getValue(), payloads);

  for (auto &&[result, resPayload] : zip_equal(getResults(), res))
    results.setMappedValues(llvm::cast<OpResult>(result), resPayload);

  return DiagnosedSilenceableFailure::success();
}
