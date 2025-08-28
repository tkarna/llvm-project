//===- DialectTransform.cpp - 'transform' dialect submodule ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/IR/DialectRegistry.h"

#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/FFIExtension/FFIExtension.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

// Global to hold the python callback handler so that it is avialable to be
// called by the C++-callback handler.
nb::object callback_handler;

void populateDialectTransformSubmodule(nb::module_ &m) {
  //===-------------------------------------------------------------------===//
  // AnyOpType
  //===-------------------------------------------------------------------===//

  auto anyOpType =
      mlir_type_subclass(m, "AnyOpType", mlirTypeIsATransformAnyOpType,
                         mlirTransformAnyOpTypeGetTypeID);
  anyOpType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirTransformAnyOpTypeGet(ctx));
      },
      "Get an instance of AnyOpType in the given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // AnyParamType
  //===-------------------------------------------------------------------===//

  auto anyParamType =
      mlir_type_subclass(m, "AnyParamType", mlirTypeIsATransformAnyParamType,
                         mlirTransformAnyParamTypeGetTypeID);
  anyParamType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirTransformAnyParamTypeGet(ctx));
      },
      "Get an instance of AnyParamType in the given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // AnyValueType
  //===-------------------------------------------------------------------===//

  auto anyValueType =
      mlir_type_subclass(m, "AnyValueType", mlirTypeIsATransformAnyValueType,
                         mlirTransformAnyValueTypeGetTypeID);
  anyValueType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirTransformAnyValueTypeGet(ctx));
      },
      "Get an instance of AnyValueType in the given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // OperationType
  //===-------------------------------------------------------------------===//

  auto operationType =
      mlir_type_subclass(m, "OperationType", mlirTypeIsATransformOperationType,
                         mlirTransformOperationTypeGetTypeID);
  operationType.def_classmethod(
      "get",
      [](nb::object cls, const std::string &operationName, MlirContext ctx) {
        MlirStringRef cOperationName =
            mlirStringRefCreate(operationName.data(), operationName.size());
        return cls(mlirTransformOperationTypeGet(ctx, cOperationName));
      },
      "Get an instance of OperationType for the given kind in the given "
      "context",
      nb::arg("cls"), nb::arg("operation_name"),
      nb::arg("context").none() = nb::none());
  operationType.def_property_readonly(
      "operation_name",
      [](MlirType type) {
        MlirStringRef operationName =
            mlirTransformOperationTypeGetOperationName(type);
        return nb::str(operationName.data, operationName.length);
      },
      "Get the name of the payload operation accepted by the handle.");

  //===-------------------------------------------------------------------===//
  // ParamType
  //===-------------------------------------------------------------------===//

  auto paramType =
      mlir_type_subclass(m, "ParamType", mlirTypeIsATransformParamType,
                         mlirTransformParamTypeGetTypeID);
  paramType.def_classmethod(
      "get",
      [](nb::object cls, MlirType type, MlirContext ctx) {
        return cls(mlirTransformParamTypeGet(ctx, type));
      },
      "Get an instance of ParamType for the given type in the given context.",
      nb::arg("cls"), nb::arg("type"), nb::arg("context").none() = nb::none());
  paramType.def_property_readonly(
      "type",
      [](MlirType type) {
        MlirType paramType = mlirTransformParamTypeGetType(type);
        return paramType;
      },
      "Get the type this ParamType is associated with.");

  nb::module_ transformFfiModule = m.def_submodule("_ffi");
  // nb::module_ transformFfiModule = m.attr("ffi");

  // transformFfiModule.def(
  //     "register_dialect_extension",
  //     [](MlirDialectRegistry wrappedRegistry) {
  //       DialectRegistry *registry = unwrap(wrappedRegistry);
  //       transform::ffi::registerDialectExtension(*registry);
  //     },
  //     "registry");

  transformFfiModule.def(
      "register_callback_handler",
      [&](nb::object callable) {
        if (callback_handler) {
          callback_handler.reset();
          transform::ffi::getHandler(nullptr);
        }

        // Mechanism to release borrow of last set callback, e.g. upon exit:
        if (callable.is_none())
          return;

        callback_handler = nb::borrow(callable);
        // NB: this borrow is only released upon another
        //     `register_callback_handler` invocation.

        // Register a C++ callback that will
        // 1) wrap its arguments,
        // 2) call a Python callback with the wrapped-up arguments,
        // 3) and unwrap the results that the Python callback returned.
        transform::ffi::getHandler(
            [&](StringRef name,
                SmallVector<SmallVector<transform::MappedValue>> args)
            -> SmallVector<SmallVector<transform::MappedValue>> {
          // Wrap up the arguments to prepare for passing them to Python.
          nb::list pyArgs;
          for (auto handleAssociatedValues : args) {
            nb::list pyAssociatedValues;

            for (auto associatedValue : handleAssociatedValues) {
              if (auto *op = dyn_cast<Operation *>(associatedValue)) {
                pyAssociatedValues.append(wrap(op));
              } else if (auto value = dyn_cast<Value>(associatedValue)) {
                pyAssociatedValues.append(wrap(value));
              } else if (auto paramAttr =
                             dyn_cast<transform::Param>(associatedValue)) {
                pyAssociatedValues.append(wrap(paramAttr));
              }
            }

            pyArgs.append(pyAssociatedValues);
          }

          // The callback to Python code.
          auto res = callback_handler(nb::str(name.data()), *pyArgs);

          // Needing to do this import here is ... not ideal.
          // The below commented-out code is potentially a better solution...
          nb::handle mlir_ir = nb::module_::import_("imex_mlir.ir");
          nb::handle Operation = mlir_ir.attr("Operation");
          nb::handle Value = mlir_ir.attr("Value");
          nb::handle Attribute = mlir_ir.attr("Attribute");

          // Unwrap the results to prepare for passing them to C++.
          SmallVector<SmallVector<transform::MappedValue>> results;
          if (nb::isinstance<nb::list>(res) || nb::isinstance<nb::tuple>(res)) {
            for (auto assocList : res) {
              SmallVector<transform::MappedValue> associatedValues;
              for (auto elt : assocList) {
                // The following is probably preferable but is broken...
                // if (nb::isinstance<MlirValue>(elt)) {
                // If `elt` is of the wrong type, isinstance call will crash.
                if (nb::isinstance(elt, Value)) {
                  auto val = nb::cast<MlirValue>(elt);
                  associatedValues.push_back(unwrap(val));
                  // The following is probably preferable but is broken...
                  //} else if (nb::isinstance<MlirOperation>(elt)) {
                  // If `elt` is of the wrong type, isinstance call will crash.
                } else if (nb::isinstance(elt, Operation)) {
                  auto op = nb::cast<MlirOperation>(elt);
                  associatedValues.push_back(unwrap(op));
                  // The following is probably preferable but is broken...
                  //} else if (nb::isinstance<MlirAttribute>(elt)) {
                  // If `elt` is of the wrong type, isinstance call will crash.
                } else if (nb::isinstance(elt, Attribute)) {
                  auto param = nb::cast<MlirAttribute>(elt);
                  associatedValues.push_back(unwrap(param));
                }
              }
              results.push_back(associatedValues);
            }
          }
          return results;
        });
      },
      nb::arg("callback").none());
}

NB_MODULE(_mlirDialectsTransform, m) {
  m.doc() = "MLIR Transform dialect.";
  populateDialectTransformSubmodule(m);
}
