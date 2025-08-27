//===- FFIExtension.cpp - FFI extension for the Transform dialect ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/FFIExtension/FFIExtension.h"
#include "mlir/Dialect/Transform/FFIExtension/FFIExtensionOps.h"
#include "mlir/IR/DialectRegistry.h"
#include <optional>

using namespace mlir;

namespace mlir {
namespace transform {
namespace ffi {
Handler &getHandler(std::optional<Handler> newHandler) {
  static Handler handler = nullptr;
  if (newHandler)
    handler = *newHandler;
  return handler;
}
} // namespace ffi
} // namespace transform
} // namespace mlir

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class FFIExtension : public transform::TransformDialectExtension<FFIExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FFIExtension)

  FFIExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/FFIExtension/FFIExtensionOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::transform::registerFFIExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<FFIExtension>();
}
