//===- FFIExtension.h - FFI extension for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_FFIEXTENSION_FFIEXTENSION_H
#define MLIR_DIALECT_TRANSFORM_FFIEXTENSION_FFIEXTENSION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class DialectRegistry;

namespace transform {
namespace ffi {

using Handler = std::function<SmallVector<SmallVector<transform::MappedValue>>(
    StringRef, SmallVector<SmallVector<transform::MappedValue>>)>;

Handler &getHandler(std::optional<Handler> newHandler = std::nullopt);

void registerDialectExtension(DialectRegistry &registry);
} // namespace ffi

/// Registers the FFI extension of the Transform dialect in the given registry.
void registerFFIExtension(DialectRegistry &dialectRegistry);
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_FFIEXTENSION_FFIEXTENSION_H
