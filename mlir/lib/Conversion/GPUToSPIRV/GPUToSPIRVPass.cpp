//===- GPUToSPIRVPass.cpp - GPU to SPIR-V Passes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert a kernel function in the GPU Dialect
// into a spirv.module operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOSPIRV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace mlirextensions {
// This op:
//   vector.create_mask %maskVal : vector<vWidth x i1>
// is lowered to:
//   if maskVal < 0
//     mask = 0
//   else if maskVal < vWidth
//     mask = (1 << maskVal) - 1
//   else
//     mask = all ones
class VectorMaskConversionPattern final
    : public mlir::OpConversionPattern<mlir::vector::CreateMaskOp> {
public:
  using OpConversionPattern<mlir::vector::CreateMaskOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp vMaskOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::VectorType vTy = vMaskOp.getVectorType();
    if (vTy.getRank() != 1)
      return mlir::failure();

    auto vWidth = vTy.getNumElements();
    assert(vWidth <= 64 && "vector.create_mask supports vector widths <= 64");
    auto vWidthConst = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(vWidth));
    auto maskVal = adaptor.getOperands()[0];
    maskVal = rewriter.create<mlir::arith::TruncIOp>(
        vMaskOp.getLoc(), rewriter.getI64Type(), maskVal);

    // maskVal < vWidth
    auto cmp = rewriter.create<mlir::arith::CmpIOp>(
        vMaskOp.getLoc(), mlir::arith::CmpIPredicate::slt, maskVal,
        vWidthConst);
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(1));
    auto shift = rewriter.create<mlir::spirv::ShiftLeftLogicalOp>(
        vMaskOp.getLoc(), one, maskVal);
    auto mask1 =
        rewriter.create<mlir::arith::SubIOp>(vMaskOp.getLoc(), shift, one);
    auto mask2 = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(-1)); // all ones
    mlir::Value sel = rewriter.create<mlir::arith::SelectOp>(vMaskOp.getLoc(),
                                                             cmp, mask1, mask2);

    // maskVal < 0
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(0));
    auto cmp2 = rewriter.create<mlir::arith::CmpIOp>(
        vMaskOp.getLoc(), mlir::arith::CmpIPredicate::slt, maskVal, zero);
    sel = rewriter.create<mlir::arith::SelectOp>(vMaskOp.getLoc(), cmp2, zero,
                                                 sel);

    sel = rewriter.create<mlir::arith::TruncIOp>(
        vMaskOp.getLoc(), rewriter.getIntegerType(vWidth), sel);
    auto res = rewriter.create<mlir::spirv::BitcastOp>(
        vMaskOp.getLoc(), mlir::VectorType::get({vWidth}, rewriter.getI1Type()),
        sel);
    vMaskOp->replaceAllUsesWith(res);
    rewriter.eraseOp(vMaskOp);
    return mlir::success();
  }
};

// This pattern converts vector.from_elements op to SPIR-V CompositeInsertOp
class VectorFromElementsConversionPattern final
    : public mlir::OpConversionPattern<mlir::vector::FromElementsOp> {
public:
  using OpConversionPattern<mlir::vector::FromElementsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::FromElementsOp fromElementsOp,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::VectorType vecTy = fromElementsOp.getType();
    if (vecTy.getRank() > 1)
      return rewriter.notifyMatchFailure(fromElementsOp,
                                         "rank > 1 vectors are not supported");

    mlir::Type spirvVecTy = getTypeConverter()->convertType(vecTy);
    if (!spirvVecTy)
      return mlir::failure();

    // if the vector is just constructed from one element
    if (mlir::isa<mlir::spirv::ScalarType>(spirvVecTy)) {
      rewriter.replaceOp(fromElementsOp, adaptor.getElements()[0]);
      return mlir::success();
    }

    auto loc = fromElementsOp.getLoc();
    mlir::Value result = rewriter.create<mlir::spirv::UndefOp>(loc, spirvVecTy);
    for (auto [idx, val] : llvm::enumerate(adaptor.getElements())) {
      result = rewriter.create<mlir::spirv::CompositeInsertOp>(loc, val, result,
                                                               idx);
    }
    rewriter.replaceOp(fromElementsOp, result);
    return mlir::success();
  }
};

// Pattern to convert arith.truncf (f32 -> bf16) followed by arith.bitcast (bf16
// -> i16) to a SPIR-V convert op.
class ArithTruncFBitcastConversionPattern final
    : public mlir::OpConversionPattern<mlir::arith::TruncFOp> {
public:
  using OpConversionPattern<mlir::arith::TruncFOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::TruncFOp truncfOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Lamda to return the element type of truncfOp
    auto getOpElementType = [&](auto op) -> mlir::Type {
      if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(op.getType()))
        return vecTy.getElementType();
      return op.getType();
    };

    if (!getOpElementType(truncfOp).isBF16())
      return mlir::failure();

    if (!truncfOp->hasOneUse())
      return mlir::failure();

    // Check if the result of truncf is used by a bitcast op
    mlir::arith::BitcastOp bitcastOp =
        mlir::dyn_cast<mlir::arith::BitcastOp>(*(truncfOp->getUsers().begin()));
    // Check if the bitcast op is converting to i16
    if (!bitcastOp || !getOpElementType(bitcastOp).isInteger(16))
      return mlir::failure();

    mlir::arith::BitcastOpAdaptor bitcastOpAdaptor(bitcastOp);

    mlir::Value intelFToBF16ConvertionOp =
        rewriter.create<mlir::spirv::INTELConvertFToBF16Op>(
            truncfOp.getLoc(),
            getTypeConverter()->convertType(bitcastOp.getType()),
            adaptor.getOperands());

    rewriter.replaceOp(bitcastOp, intelFToBF16ConvertionOp);
    rewriter.eraseOp(truncfOp);
    return mlir::success();
  }
};

// Pattern to convert arith.bitcast (i16 -> bf16) followed by arith.extf (bf16
// -> f32) to a SPIR-V convert op.
class ArithBitcastExtFConversionPattern final
    : public mlir::OpConversionPattern<mlir::arith::BitcastOp> {
public:
  using OpConversionPattern<mlir::arith::BitcastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::BitcastOp bitcastOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Lamda to return the element type of bitcsatOp
    auto getOpElementType = [&](auto op) -> mlir::Type {
      if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(op.getType()))
        return vecTy.getElementType();
      return op.getType();
    };

    if (!getOpElementType(bitcastOp).isBF16())
      return mlir::failure();

    if (!bitcastOp->hasOneUse())
      return mlir::failure();

    // Check if the result of bitcast is used by an extf op
    mlir::arith::ExtFOp extfOp =
        mlir::dyn_cast<mlir::arith::ExtFOp>(*(bitcastOp->getUsers().begin()));
    // Check if the extf op is converting to f32
    if (!extfOp || !getOpElementType(extfOp).isF32())
      return mlir::failure();

    mlir::Value intelBF16ToFConvertionOp =
        rewriter.create<mlir::spirv::INTELConvertBF16ToFOp>(
            bitcastOp.getLoc(),
            getTypeConverter()->convertType(extfOp.getType()),
            adaptor.getOperands());

    rewriter.replaceOp(extfOp, intelBF16ToFConvertionOp);
    rewriter.eraseOp(bitcastOp);

    return mlir::success();
  }
};

void populateBF16ArithToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.add<ArithTruncFBitcastConversionPattern,
               ArithBitcastExtFConversionPattern>(typeConverter,
                                                  patterns.getContext());
}

void populateVectorToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                   mlir::RewritePatternSet &patterns) {
  patterns
      .add<VectorFromElementsConversionPattern, VectorMaskConversionPattern>(
          typeConverter, patterns.getContext());
}

} // namespace mlirextensions

namespace {
/// Pass to lower GPU Dialect to SPIR-V. The pass only converts the gpu.func ops
/// inside gpu.module ops. i.e., the function that are referenced in
/// gpu.launch_func ops. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
struct GPUToSPIRVPass final : impl::ConvertGPUToSPIRVBase<GPUToSPIRVPass> {
  explicit GPUToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  /// Queries the target environment from 'targets' attribute of the given
  /// `moduleOp`.
  spirv::TargetEnvAttr lookupTargetEnvInTargets(gpu::GPUModuleOp moduleOp);

  /// Queries the target environment from 'targets' attribute of the given
  /// `moduleOp` or returns target environment as returned by
  /// `spirv::lookupTargetEnvOrDefault` if not provided by 'targets'.
  spirv::TargetEnvAttr lookupTargetEnvOrDefault(gpu::GPUModuleOp moduleOp);
  bool mapMemorySpace;
};

spirv::TargetEnvAttr
GPUToSPIRVPass::lookupTargetEnvInTargets(gpu::GPUModuleOp moduleOp) {
  if (ArrayAttr targets = moduleOp.getTargetsAttr()) {
    for (Attribute targetAttr : targets)
      if (auto spirvTargetEnvAttr = dyn_cast<spirv::TargetEnvAttr>(targetAttr))
        return spirvTargetEnvAttr;
  }

  return {};
}

spirv::TargetEnvAttr
GPUToSPIRVPass::lookupTargetEnvOrDefault(gpu::GPUModuleOp moduleOp) {
  if (spirv::TargetEnvAttr targetEnvAttr = lookupTargetEnvInTargets(moduleOp))
    return targetEnvAttr;

  return spirv::lookupTargetEnvOrDefault(moduleOp);
}

void GPUToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  SmallVector<Operation *, 1> gpuModules;
  OpBuilder builder(context);

  auto targetEnvSupportsKernelCapability = [this](gpu::GPUModuleOp moduleOp) {
    auto targetAttr = lookupTargetEnvOrDefault(moduleOp);
    spirv::TargetEnv targetEnv(targetAttr);
    return targetEnv.allows(spirv::Capability::Kernel);
  };

  module.walk([&](gpu::GPUModuleOp moduleOp) {
    // Clone each GPU kernel module for conversion, given that the GPU
    // launch op still needs the original GPU kernel module.
    // For Vulkan Shader capabilities, we insert the newly converted SPIR-V
    // module right after the original GPU module, as that's the expectation of
    // the in-tree SPIR-V CPU runner (the Vulkan runner does not use this pass).
    // For OpenCL Kernel capabilities, we insert the newly converted SPIR-V
    // module inside the original GPU module, as that's the expectaion of the
    // normal GPU compilation pipeline.
    if (targetEnvSupportsKernelCapability(moduleOp)) {
      builder.setInsertionPointToStart(moduleOp.getBody());
    } else {
      builder.setInsertionPoint(moduleOp.getOperation());
    }
    gpuModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  // Run conversion for each module independently as they can have different
  // TargetEnv attributes.
  for (Operation *gpuModule : gpuModules) {
    spirv::TargetEnvAttr targetAttr =
        lookupTargetEnvOrDefault(cast<gpu::GPUModuleOp>(gpuModule));

    // Map MemRef memory space to SPIR-V storage class first if requested.
    if (mapMemorySpace) {
      spirv::MemorySpaceToStorageClassMap memorySpaceMap =
          targetEnvSupportsKernelCapability(
              dyn_cast<gpu::GPUModuleOp>(gpuModule))
              ? spirv::mapMemorySpaceToOpenCLStorageClass
              : spirv::mapMemorySpaceToVulkanStorageClass;
      spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);
      spirv::convertMemRefTypesAndAttrs(gpuModule, converter);

      // Check if there are any illegal ops remaining.
      std::unique_ptr<ConversionTarget> target =
          spirv::getMemorySpaceToStorageClassTarget(*context);
      gpuModule->walk([&target, this](Operation *childOp) {
        if (target->isIllegal(childOp)) {
          childOp->emitOpError("failed to legalize memory space");
          signalPassFailure();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    options.use64bitIndex = this->use64bitIndex;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    // Upstream SPIRVTypeConverter does not add conversion for
    // UnrankedMemRefType.
    // Conversion logic is the same as ranked dynamic memref type for OpenCL
    // Kernel. unranked memref type is converted to a spirv pointer type
    // with converted spirv scalar element type and spirv storage class.
    // Only scalar element type is currently supported.
    // Also vulkan should be handled differently but out of scope since this
    // conversion pass is for lowering to OpenCL spirv kernel only.
    typeConverter.addConversion(
        [&](mlir::UnrankedMemRefType type) -> std::optional<mlir::Type> {
          auto attr = mlir::dyn_cast_or_null<mlir::spirv::StorageClassAttr>(
              type.getMemorySpace());
          if (!attr)
            return nullptr;
          mlir::spirv::StorageClass storageClass = attr.getValue();

          mlir::Type elementType = type.getElementType();
          auto scalarType =
              mlir::dyn_cast<mlir::spirv::ScalarType>(elementType);
          if (!scalarType)
            return nullptr;
          mlir::Type arrayElemType = typeConverter.convertType(scalarType);
          return mlir::spirv::PointerType::get(arrayElemType, storageClass);
        });

    populateMMAToSPIRVCoopMatrixTypeConversion(typeConverter);

    RewritePatternSet patterns(context);
    mlirextensions::populateBF16ArithToSPIRVPatterns(typeConverter, patterns);
    mlirextensions::populateVectorToSPIRVPatterns(typeConverter, patterns);

    populateGPUToSPIRVPatterns(typeConverter, patterns);
    populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(typeConverter,
                                                          patterns);

    // TODO: Change SPIR-V conversion to be progressive and remove the following
    // patterns.
    ScfToSPIRVContext scfContext;
    populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    populateMemRefToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);
    populateVectorToSPIRVPatterns(typeConverter, patterns);

    mlir::populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
    mlir::populateComplexToSPIRVPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    mlir::index::populateIndexToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);
    mlir::populateTensorToSPIRVPatterns(typeConverter,
                                        /*byteCountThreshold=*/64, patterns);
    mlir::ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
      return signalPassFailure();
  }

  // For OpenCL, the gpu.func op in the original gpu.module op needs to be
  // replaced with an empty func.func op with the same arguments as the gpu.func
  // op. The func.func op needs gpu.kernel attribute set.
  module.walk([&](gpu::GPUModuleOp moduleOp) {
    if (targetEnvSupportsKernelCapability(moduleOp)) {
      moduleOp.walk([&](gpu::GPUFuncOp funcOp) {
        builder.setInsertionPoint(funcOp);
        auto newFuncOp =
            func::FuncOp::create(builder, funcOp.getLoc(), funcOp.getName(),
                                 funcOp.getFunctionType());
        auto entryBlock = newFuncOp.addEntryBlock();
        builder.setInsertionPointToEnd(entryBlock);
        func::ReturnOp::create(builder, funcOp.getLoc());
        newFuncOp->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           builder.getUnitAttr());
        funcOp.erase();
      });
    }
  });
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertGPUToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUToSPIRVPass>(mapMemorySpace);
}
