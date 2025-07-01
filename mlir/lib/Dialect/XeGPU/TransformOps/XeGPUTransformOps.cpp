//===- XeGPUTransformOps.cpp - Implementation of XeGPU transformation ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/Utils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <numeric>

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "xegpu-transforms"

using namespace mlir;
using namespace mlir::transform;

class XeGPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          XeGPUTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XeGPUTransformDialectExtension)

  using Base::Base;

  void init();
};

void XeGPUTransformDialectExtension::init() {
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<gpu::GPUDialect>();
  declareGeneratedDialect<xegpu::XeGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include <mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.cpp.inc>
      >();
}

#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.cpp.inc>

void mlir::xegpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<XeGPUTransformDialectExtension>();
}

/// Assuming that `ofr` is an index attr or a param of index type
/// or a transform dialect handle mapped to exactly one op
/// with one index result, get that value and cast it to int type.
static DiagnosedSilenceableFailure convertMixedValuesToInt(
    transform::TransformState &state, TransformOpInterface transformOp,
    SmallVectorImpl<int32_t> &result, ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    // Attribute case.
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        result.push_back(intAttr.getInt());
      } else {
        return transformOp.emitDefiniteFailure() << "expected IntegerAttr";
      }
      continue;
    }

    // Transform param case.
    Value transformValue = cast<Value>(ofr);
    if (isa<TransformParamTypeInterface>(transformValue.getType())) {
      ArrayRef<Attribute> params = state.getParams(transformValue);
      if (params.size() != 1)
        return transformOp.emitDefiniteFailure()
               << "requires exactly one parameter associated";
      result.push_back(
          cast<IntegerAttr>(params.front()).getValue().getSExtValue());
      continue;
    }

    // Payload value case.
    auto payloadOps = state.getPayloadOps(transformValue);
    if (!llvm::hasSingleElement(payloadOps)) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "handle must be mapped to exactly one payload op";
      diag.attachNote(transformValue.getLoc())
          << "mapped to " << llvm::range_size(payloadOps) << " payload ops";
      return diag;
    }

    Operation *op = *payloadOps.begin();
    if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex()) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "payload op must have exactly 1 index result";
      diag.attachNote(op->getLoc())
          << "has " << op->getNumResults() << " results";
      return diag;
    }

    IntegerAttr intAttr;
    if (!matchPattern(op->getResult(0), m_Constant(&intAttr)))
      return transformOp.emitSilenceableError()
             << "requires param or handle to be the result of a constant like "
                "op";

    result.push_back(intAttr.getInt());
  }
  return DiagnosedSilenceableFailure::success();
}

/// Recurse operands and collect all producer ops in the given region.
void collectProducerOps(Operation *op, Region &inRegion,
                        SmallVector<Operation *> &ops) {
  for (auto val : op->getOperands()) {
    if (const auto definingOp = val.getDefiningOp();
        definingOp && definingOp->getParentRegion() == &inRegion) {
      ops.push_back(definingOp);
      collectProducerOps(definingOp, inRegion, ops);
    }
  }
}

/// Returns all producer ops in the given region
SmallVector<Operation *> getProducerOpsInRegion(Operation *op, Region &inRegion,
                                                bool includeOp = true) {
  SmallVector<Operation *> producerOps;
  if (includeOp) {
    producerOps.push_back(op);
  }
  collectProducerOps(op, inRegion, producerOps);
  return producerOps;
}

/// Find xegpu.create_nd_desc op for the given operand value.
static std::optional<xegpu::CreateNdDescOp>
findDescriptorOp(Value operandValue, Operation *userOp) {
  // FIXME more generic way of finding desc op that may be outside the loop
  Value currentValue = operandValue;
  if (!currentValue.getDefiningOp()) {
    // Desc op may reside outside a loop.
    auto forOp = userOp->getParentOfType<LoopLikeOpInterface>();
    if (!forOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to find operand desc op, def op not a loop.");
      return std::nullopt;
    }
    int64_t iterArgIdx;
    if (auto iterArg = llvm::dyn_cast<BlockArgument>(currentValue)) {
      auto numInductionVars = forOp.getLoopInductionVars()->size();
      iterArgIdx = iterArg.getArgNumber() - numInductionVars;
      currentValue = forOp.getInits()[iterArgIdx];
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to find operand desc op, def op not an init val.");
      return std::nullopt;
    }
  }
  auto findDescOp = [](Value val) -> std::optional<xegpu::CreateNdDescOp> {
    Operation *producerOp = val.getDefiningOp();
    while (producerOp) {
      if (auto maybeDescOp = dyn_cast<xegpu::CreateNdDescOp>(producerOp)) {
        return maybeDescOp;
      }
      if (producerOp->getNumOperands() == 0)
        break;
      producerOp = producerOp->getOperand(0).getDefiningOp();
    }
    return std::nullopt;
  };
  return findDescOp(currentValue);
}

// Get user of type T in immediate users of the value.
template <typename T>
static std::optional<T> getUserOfType(Value value) {
  auto users = value.getUsers();
  auto it = llvm::find_if(users, [&](Operation *op) { return isa<T>(op); });
  if (it != users.end()) {
    return cast<T>(*it);
  }
  return std::nullopt;
}

/// Create a layout attribute from the given parameters.
xegpu::LayoutAttr createLayoutAttr(MLIRContext *ctx, ArrayRef<int32_t> sgLayout,
                                   ArrayRef<int32_t> sgData,
                                   std::optional<ArrayRef<int32_t>> instData) {
  return xegpu::LayoutAttr::get(
      ctx, DenseI32ArrayAttr::get(ctx, sgLayout),
      DenseI32ArrayAttr::get(ctx, sgData),
      instData ? DenseI32ArrayAttr::get(ctx, instData.value()) : nullptr,
      /*lane_layout=*/nullptr,
      /*lane_data=*/nullptr,
      /*order=*/nullptr);
}

/// Replace xegpu.create_nd_desc op with a new one with the given layout.
xegpu::CreateNdDescOp setDescLayout(transform::TransformRewriter &rewriter,
                                    xegpu::CreateNdDescOp descOp,
                                    xegpu::LayoutAttr layout) {
  auto oldTensorDesc = descOp.getResult();
  auto descShapedType = cast<ShapedType>(oldTensorDesc.getType());
  // TODO inherit desc attributes from old op (if any)
  auto descType = xegpu::TensorDescType::get(
      descShapedType.getShape(), descShapedType.getElementType(),
      /*array_length=*/1,
      /*boundary_check=*/true,
      /*memory_space=*/xegpu::MemorySpace::Global,
      /*layout=*/layout);

  rewriter.setInsertionPointAfter(descOp);
  if (descOp.getMixedOffsets().size() > 0) {
    auto newDescOp = rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
        descOp, descType, descOp.getSource(), descOp.getMixedOffsets(),
        descOp.getMixedSizes(), descOp.getMixedStrides());
    return newDescOp;
  }
  auto newDescOp = rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descType, descOp.getSource(), descOp.getMixedSizes(),
      descOp.getMixedStrides());
  return newDescOp;
}

void transform::InsertPrefetchOp::build(OpBuilder &builder,
                                        OperationState &ostate, Value target,
                                        Value loop, int64_t operandIndex,
                                        int64_t nbPrefetch,
                                        ArrayRef<OpFoldResult> mixedSgLayout,
                                        ArrayRef<OpFoldResult> mixedSgData,
                                        ArrayRef<OpFoldResult> mixedInstData) {
  SmallVector<int64_t> staticSgLayout, staticSgData, staticInstData;
  SmallVector<Value> dynamicSgLayout, dynamicSgData, dynamicInstData;
  dispatchIndexOpFoldResults(mixedSgLayout, dynamicSgLayout, staticSgLayout);
  dispatchIndexOpFoldResults(mixedSgData, dynamicSgData, staticSgData);
  dispatchIndexOpFoldResults(mixedInstData, dynamicInstData, staticInstData);
  SmallVector<Type> resultTypes{target.getType(), loop.getType()};
  build(builder, ostate,
        /*resultTypes=*/resultTypes,
        /*target=*/target,
        /*loop=*/loop,
        /*operandIndex=*/operandIndex,
        /*dynamic_nb_prefetch=*/nullptr,
        /*sg_layout=*/dynamicSgLayout,
        /*sg_data=*/dynamicSgData,
        /*inst_data=*/dynamicInstData,
        /*static_nb_prefetch=*/nbPrefetch,
        /*static_sg_layout=*/staticSgLayout,
        /*static_sg_data=*/staticSgData,
        /*static_inst_data=*/staticInstData);
}

DiagnosedSilenceableFailure
transform::InsertPrefetchOp::apply(transform::TransformRewriter &rewriter,
                                   transform::TransformResults &results,
                                   transform::TransformState &state) {

  auto targetOps = state.getPayloadOps(getTarget());
  auto loopOps = state.getPayloadOps(getLoop());

  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  if (!llvm::hasSingleElement(loopOps)) {
    return emitDefiniteFailure() << "requires exactly one loopOp handle (got "
                                 << llvm::range_size(loopOps) << ")";
  }

  Operation *targetPtr = *targetOps.begin();
  // For now only DPAS op is supported.
  auto targetOp = dyn_cast<xegpu::DpasOp>(targetPtr);
  if (!targetOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a xegpu.dpas op, but got: " << targetPtr->getName();
  }

  Operation *loopPtr = *loopOps.begin();
  auto forOp = dyn_cast<scf::ForOp>(loopPtr);
  if (!forOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a scf.for op, but got: " << loopPtr->getName();
  }

  auto parentLoop = targetOp->getParentOfType<scf::ForOp>();
  if (!parentLoop || parentLoop != forOp) {
    return emitSilenceableFailure(getLoc())
           << "target op is not contained in the given scf.for loop.";
  }

  int64_t operandIndex = getOperandIndex();
  if (operandIndex >= targetOp.getNumOperands()) {
    return emitSilenceableFailure(getLoc())
           << "operandIndex exceeds the number of op operands.";
  }

  auto transformOp = cast<TransformOpInterface>(getOperation());

  SmallVector<int32_t> sgLayout;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, transformOp, sgLayout, getMixedSgLayout());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> sgData;
  status =
      convertMixedValuesToInt(state, transformOp, sgData, getMixedSgData());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> instData;
  status =
      convertMixedValuesToInt(state, transformOp, instData, getMixedInstData());
  if (!status.succeeded())
    return status;

  if (sgLayout.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_layout to be a 2D vector";
  }
  if (sgData.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_data to be a 2D vector";
  }

  int64_t nbPrefetch = getStaticNbPrefetch();
  if (getDynamicNbPrefetch()) {
    // Get dynamic prefetch count from transform param or handle.
    SmallVector<int32_t> dynamicNbPrefetch;
    status = convertMixedValuesToInt(state, transformOp, dynamicNbPrefetch,
                                     {getDynamicNbPrefetch()});
    if (!status.succeeded())
      return status;
    if (dynamicNbPrefetch.size() != 1) {
      return emitDefiniteFailure()
             << "requires exactly one value for dynamic_nb_prefetch";
    }
    nbPrefetch = dynamicNbPrefetch[0];
  }
  if (nbPrefetch <= 0) {
    return emitSilenceableFailure(getLoc())
           << "nb_prefetch must be a positive integer.";
  }

  // Find load operation of the operand.
  Value opVec = targetOp.getOperation()->getOperand(operandIndex);
  auto defOp = opVec.getDefiningOp();
  if (!defOp) {
    return emitSilenceableFailure(getLoc())
           << "Could not find defining op of the operand.";
  }
  auto producers = getProducerOpsInRegion(defOp, parentLoop.getRegion(), true);
  Operation *maybeLoadOp = nullptr;
  for (auto &op : llvm::reverse(producers)) {
    if (isa<xegpu::LoadNdOp>(op)) {
      maybeLoadOp = op;
      break;
    }
  }
  if (!maybeLoadOp) {
    return emitSilenceableFailure(getLoc()) << "Could not find load op.";
  }
  auto loadOp = cast<xegpu::LoadNdOp>(maybeLoadOp);
  if (!loadOp.getConstOffsets()) {
    return emitSilenceableFailure(getLoc())
           << "load op must have constant offsets.";
  }

  // Find descriptor op.
  auto maybeDescOp = findDescriptorOp(opVec, targetOp.getOperation());
  if (!maybeDescOp) {
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  }
  auto descOp = *maybeDescOp;
  if (descOp.getMixedOffsets().size() > 0) {
    return emitSilenceableFailure(getLoc())
           << "desc op with offsets is not supported.";
  }

  // Clone desc op outside the loop.
  rewriter.setInsertionPoint(forOp);
  auto newDescOp =
      cast<xegpu::CreateNdDescOp>(rewriter.clone(*descOp.getOperation()));
  // Set desc op layout.
  auto layout =
      createLayoutAttr(rewriter.getContext(), sgLayout, sgData, instData);
  newDescOp = setDescLayout(rewriter, newDescOp, layout);

  // Clone reduction loop for initial prefetches.
  // Compute upper bound of the loop.
  auto nbPrefetchCst =
      rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), nbPrefetch);
  auto nbStep = rewriter.createOrFold<arith::MulIOp>(
      forOp.getLoc(), nbPrefetchCst, forOp.getStep());
  auto initUpBound = rewriter.createOrFold<arith::AddIOp>(
      forOp.getLoc(), forOp.getLowerBound(), nbStep);
  auto initForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), initUpBound, forOp.getStep());

  auto ctx = rewriter.getContext();
  auto readCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);

  // Replace loop induction variable in offsets with the given value.
  auto getPrefetchOffsets =
      [&](Value indVarReplacement) -> SmallVector<OpFoldResult> {
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), indVarReplacement);
    SmallVector<Value> dynamicOffsets =
        llvm::to_vector(llvm::map_range(loadOp.getOffsets(), [&](Value v) {
          return mapping.lookupOrDefault(v);
        }));
    auto constOffsets = loadOp.getConstOffsets().value();
    return getMixedValues(constOffsets, dynamicOffsets, ctx);
  };

  // Insert prefetch op in init loop.
  // Replace induction var with the init loop induction var.
  rewriter.setInsertionPointToStart(initForOp.getBody());
  rewriter.create<xegpu::PrefetchNdOp>(
      newDescOp.getLoc(), newDescOp.getResult(),
      getPrefetchOffsets(initForOp.getInductionVar()), readCacheHint,
      readCacheHint, readCacheHint);

  // Insert prefetch op in main loop.
  // Calculate prefetch offset
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto prefetchOffset = rewriter.create<arith::AddIOp>(
      forOp.getLoc(), forOp.getInductionVar(), nbStep);
  // Replace induction var with correct offset.
  rewriter.create<xegpu::PrefetchNdOp>(
      newDescOp.getLoc(), newDescOp.getResult(),
      getPrefetchOffsets(prefetchOffset), readCacheHint, readCacheHint,
      readCacheHint);

  // Unroll the init loop.
  if (failed(loopUnrollFull(initForOp))) {
    return emitSilenceableFailure(getLoc()) << "Failed to unroll the loop";
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::InsertPrefetchOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getLoopMutable(), effects);
  onlyReadsHandle(getDynamicNbPrefetchMutable(), effects);
  onlyReadsHandle(getSgLayoutMutable(), effects);
  onlyReadsHandle(getSgDataMutable(), effects);
  onlyReadsHandle(getInstDataMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::GetDescOp::applyToOne(transform::TransformRewriter &rewriter,
                                 Operation *target,
                                 transform::ApplyToEachResultList &results,
                                 transform::TransformState &state) {

  int64_t operandIndex = getOperandIndex();
  if (operandIndex >= target->getNumOperands()) {
    return emitSilenceableFailure(getLoc())
           << "operandIndex exceeds the number of op operands.";
  }

  Value opVec = target->getOperand(operandIndex);
  auto maybeDescOp = findDescriptorOp(opVec, target);
  if (!maybeDescOp) {
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  }

  results.push_back(*maybeDescOp);
  return DiagnosedSilenceableFailure::success();
}

void transform::GetDescOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void transform::SetDescLayoutOp::build(OpBuilder &builder,
                                       OperationState &result, Value target,
                                       int64_t resultIndex,
                                       ArrayRef<OpFoldResult> mixedSgLayout,
                                       ArrayRef<OpFoldResult> mixedSgData,
                                       ArrayRef<OpFoldResult> mixedInstData) {
  SmallVector<int64_t> staticSgLayout, staticSgData, staticInstData;
  SmallVector<Value> dynamicSgLayout, dynamicSgData, dynamicInstData;
  dispatchIndexOpFoldResults(mixedSgLayout, dynamicSgLayout, staticSgLayout);
  dispatchIndexOpFoldResults(mixedSgData, dynamicSgData, staticSgData);
  dispatchIndexOpFoldResults(mixedInstData, dynamicInstData, staticInstData);
  build(builder, result, target.getType(),
        /*target=*/target,
        /*resultIndex=*/resultIndex,
        /*sg_layout=*/dynamicSgLayout,
        /*sg_data=*/dynamicSgData,
        /*inst_data=*/dynamicInstData,
        /*static_sg_layout=*/staticSgLayout,
        /*static_sg_data=*/staticSgData,
        /*static_inst_data=*/staticInstData);
}

DiagnosedSilenceableFailure
transform::SetDescLayoutOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {

  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  auto transformOp = cast<TransformOpInterface>(getOperation());

  SmallVector<int32_t> sgLayout;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, transformOp, sgLayout, getMixedSgLayout());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> sgData;
  status =
      convertMixedValuesToInt(state, transformOp, sgData, getMixedSgData());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> instData;
  status =
      convertMixedValuesToInt(state, transformOp, instData, getMixedInstData());
  if (!status.succeeded())
    return status;

  // For now only create_nd_desc op is supported.
  auto descOp = dyn_cast<xegpu::CreateNdDescOp>(target);
  if (!descOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a xegpu.create_nd_desc op, but got: "
                << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  // Set layout attr in desc op's return type. Replaces old desc op.
  auto layoutAttr =
      createLayoutAttr(rewriter.getContext(), sgLayout, sgData, instData);
  auto newdescOp = setDescLayout(rewriter, descOp, layoutAttr);

  // Map result handles.
  results.set(cast<OpResult>(getTransformed()), {newdescOp.getOperation()});

  return DiagnosedSilenceableFailure::success();
}

void transform::SetDescLayoutOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getSgLayoutMutable(), effects);
  onlyReadsHandle(getSgDataMutable(), effects);
  onlyReadsHandle(getInstDataMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void transform::SetOpLayoutAttrOp::build(
    OpBuilder &builder, OperationState &ostate, Value target, int64_t index,
    ArrayRef<OpFoldResult> mixedSgLayout, ArrayRef<OpFoldResult> mixedSgData,
    ArrayRef<OpFoldResult> mixedInstData, bool result) {
  SmallVector<int64_t> staticSgLayout, staticSgData, staticInstData;
  SmallVector<Value> dynamicSgLayout, dynamicSgData, dynamicInstData;
  dispatchIndexOpFoldResults(mixedSgLayout, dynamicSgLayout, staticSgLayout);
  dispatchIndexOpFoldResults(mixedSgData, dynamicSgData, staticSgData);
  dispatchIndexOpFoldResults(mixedInstData, dynamicInstData, staticInstData);
  build(builder, ostate, target.getType(),
        /*target=*/target,
        /*index=*/index,
        /*sg_layout=*/dynamicSgLayout,
        /*sg_data=*/dynamicSgData,
        /*inst_data=*/dynamicInstData,
        /*static_sg_layout=*/staticSgLayout,
        /*static_sg_data=*/staticSgData,
        /*static_inst_data=*/staticInstData,
        /*result=*/result);
}

DiagnosedSilenceableFailure
transform::SetOpLayoutAttrOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {

  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  bool resultTarget = getResult();

  int64_t index = getIndex();
  if (resultTarget && index >= target->getNumResults()) {
    return emitSilenceableFailure(getLoc())
           << "index exceeds the number of op results.";
  }
  if (!resultTarget && index >= target->getNumOperands()) {
    return emitSilenceableFailure(getLoc())
           << "index exceeds the number of op operands.";
  }

  auto transformOp = cast<TransformOpInterface>(getOperation());

  SmallVector<int32_t> sgLayout;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, transformOp, sgLayout, getMixedSgLayout());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> sgData;
  status =
      convertMixedValuesToInt(state, transformOp, sgData, getMixedSgData());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> instData;
  status =
      convertMixedValuesToInt(state, transformOp, instData, getMixedInstData());
  if (!status.succeeded())
    return status;

  auto layoutAttr =
      createLayoutAttr(rewriter.getContext(), sgLayout, sgData, instData);
  // Set layout attribute for the op result or operand
  if (resultTarget) {
    xegpu::setDistributeLayoutAttr(target->getResult(index), layoutAttr);
  } else {
    xegpu::setDistributeLayoutAttr(target->getOpOperand(index), layoutAttr);
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::SetOpLayoutAttrOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getSgLayoutMutable(), effects);
  onlyReadsHandle(getSgDataMutable(), effects);
  onlyReadsHandle(getInstDataMutable(), effects);
  modifiesPayload(effects);
}

void transform::ConvertOperandLayoutOp::build(
    OpBuilder &builder, OperationState &ostate, Value target, int64_t index,
    ArrayRef<OpFoldResult> mixedSgLayout, ArrayRef<OpFoldResult> mixedSgData,
    ArrayRef<OpFoldResult> mixedInstData) {
  SmallVector<int64_t> staticSgLayout, staticSgData, staticInstData;
  SmallVector<Value> dynamicSgLayout, dynamicSgData, dynamicInstData;
  dispatchIndexOpFoldResults(mixedSgLayout, dynamicSgLayout, staticSgLayout);
  dispatchIndexOpFoldResults(mixedSgData, dynamicSgData, staticSgData);
  dispatchIndexOpFoldResults(mixedInstData, dynamicInstData, staticInstData);
  build(builder, ostate, target.getType(),
        /*target=*/target,
        /*index=*/index,
        /*sg_layout=*/dynamicSgLayout,
        /*sg_data=*/dynamicSgData,
        /*inst_data=*/dynamicInstData,
        /*static_sg_layout=*/staticSgLayout,
        /*static_sg_data=*/staticSgData,
        /*static_inst_data=*/staticInstData);
}

DiagnosedSilenceableFailure
transform::ConvertOperandLayoutOp::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {

  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  // For now only DPAS op is supported.
  auto targetOp = dyn_cast<xegpu::DpasOp>(target);
  if (!targetOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a xegpu.dpas op, but got: " << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  int64_t operandIndex = getOperandIndex();
  if (operandIndex >= targetOp.getNumOperands()) {
    return emitSilenceableFailure(getLoc())
           << "operandIndex exceeds the number of op operands.";
  }

  auto transformOp = cast<TransformOpInterface>(getOperation());

  SmallVector<int32_t> sgLayout;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, transformOp, sgLayout, getMixedSgLayout());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> sgData;
  status =
      convertMixedValuesToInt(state, transformOp, sgData, getMixedSgData());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> instData;
  status =
      convertMixedValuesToInt(state, transformOp, instData, getMixedInstData());
  if (!status.succeeded())
    return status;

  if (sgLayout.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_layout to be a 2D vector";
  }
  if (sgData.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_data to be a 2D vector";
  }
  if (instData.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected inst_data to be a 2D vector";
  }

  // Find desc op.
  Value opVec = target->getOperand(operandIndex);
  auto maybeDescOp = findDescriptorOp(opVec, targetOp.getOperation());
  if (!maybeDescOp) {
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  }
  auto descOp = *maybeDescOp;
  // Get load op.
  auto maybeLoadOp = getUserOfType<xegpu::LoadNdOp>(descOp.getResult());
  if (!maybeLoadOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a xegpu.load_nd op as a user of the descriptor op.";
  }
  auto loadOp = *maybeLoadOp;
  // Get load op operand value layout
  auto producerLayoutAttr =
      xegpu::getDistributeLayoutAttr(loadOp.getOperand(0));
  if (!producerLayoutAttr) {
    return emitSilenceableFailure(getLoc())
           << "Operand producer op does not have a layout attr.";
  }

  // New layout attr
  auto layoutAttr =
      createLayoutAttr(rewriter.getContext(), sgLayout, sgData, instData);

  if (producerLayoutAttr != layoutAttr) {
    rewriter.setInsertionPointAfter(loadOp.getOperation());
    auto source = loadOp.getResult();
    auto convLayoutOp = rewriter.create<xegpu::ConvertLayoutOp>(
        loadOp.getLoc(), source.getType(), source, producerLayoutAttr,
        layoutAttr);
    // Replace load op result with the converted layout.
    rewriter.replaceUsesWithIf(
        source, convLayoutOp.getResult(), [&](OpOperand &use) {
          return use.getOwner() != convLayoutOp.getOperation();
        });
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::ConvertOperandLayoutOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getSgLayoutMutable(), effects);
  onlyReadsHandle(getSgDataMutable(), effects);
  onlyReadsHandle(getInstDataMutable(), effects);
  modifiesPayload(effects);
}

void transform::SetGPULaunchThreadsOp::build(
    OpBuilder &builder, OperationState &ostate, Value target,
    ArrayRef<OpFoldResult> mixedThreads) {
  SmallVector<int64_t> staticThreads;
  SmallVector<Value> dynamicThreads;
  dispatchIndexOpFoldResults(mixedThreads, dynamicThreads, staticThreads);
  build(builder, ostate, target.getType(),
        /*target=*/target,
        /*threads=*/dynamicThreads,
        /*static_threads=*/staticThreads);
}

DiagnosedSilenceableFailure
transform::SetGPULaunchThreadsOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {

  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  auto launchOp = dyn_cast<gpu::LaunchOp>(target);
  if (!launchOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a gpu.launch op, but got: " << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  auto transformOp = cast<TransformOpInterface>(getOperation());

  SmallVector<int32_t> threads;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, transformOp, threads, getMixedThreads());
  if (!status.succeeded())
    return status;

  if (threads.size() != 3) {
    return emitSilenceableFailure(getLoc())
           << "Expected threads to be a 3D vector";
  }

  rewriter.setInsertionPoint(launchOp);
  auto createConstValue = [&](int value) {
    return rewriter.create<arith::ConstantIndexOp>(launchOp.getLoc(), value);
  };

  // Replace threads in-place.
  launchOp.getBlockSizeXMutable().assign(createConstValue(threads[0]));
  launchOp.getBlockSizeYMutable().assign(createConstValue(threads[1]));
  launchOp.getBlockSizeZMutable().assign(createConstValue(threads[2]));

  return DiagnosedSilenceableFailure::success();
}

void transform::SetGPULaunchThreadsOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getThreadsMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::ExpandResultVectorOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {

  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  // Check that target is a vector.transfer_read op.
  if (!isa<vector::TransferReadOp>(target)) {
    return emitDefiniteFailure()
           << "expected a vector.transfer_read op, but got: "
           << target->getName();
  }
  auto readOp = dyn_cast<vector::TransferReadOp>(target);

  // Replace transfer_read op with new op whose return vector's dimension
  // has been extended by a singleton dim in the leading dimension.
  auto vecType = cast<VectorType>(target->getResult(0).getType());
  auto oldShape = vecType.getShape();
  SmallVector<int64_t> newShape{1};
  newShape.append(oldShape.begin(), oldShape.end());
  auto newType = VectorType::get(newShape, vecType.getElementType());
  rewriter.setInsertionPointAfter(readOp);
  // TODO clone read op retaining attributes (if any)
  auto inBounds = SmallVector<bool>{true, true};
  auto newOp = rewriter.create<vector::TransferReadOp>(
      target->getLoc(), newType, readOp.getBase(),
      ValueRange{readOp.getIndices()}, std::nullopt, inBounds);
  rewriter.replaceOp(target, newOp);

  // Map result handles.
  results.set(cast<OpResult>(getTransformed()), {newOp.getOperation()});

  return DiagnosedSilenceableFailure::success();
}

void transform::ExpandResultVectorOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
