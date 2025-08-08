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
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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

/// Add offset update op after create desc op if tile is updated in the loop.
xegpu::CreateNdDescOp insertUpdateOp(transform::TransformRewriter &rewriter,
                                     scf::ForOp parentLoopOp,
                                     xegpu::CreateNdDescOp descOp) {
  // DescOp offset is an affine map with loop dependent and independent
  // components. The new desc op will be loop independent, i.e. it uses the
  // constant offset. The remainder offset, 'offset - constant', will be used
  // in the update offset op.

  // Compute the constant offset.
  // Clone desc op producers and replace loop variable with lower bound.
  rewriter.setInsertionPointAfter(descOp);
  auto loc = descOp.getLoc();
  IRMapping mapping;
  SmallVector<Operation *> clonedOps;
  auto producers = getProducerOpsInRegion(descOp.getOperation(),
                                          parentLoopOp.getRegion(), true);
  for (auto &op : llvm::reverse(producers)) {
    auto newOp = rewriter.clone(*op, mapping);
    clonedOps.push_back(newOp);
  }
  // Replace loop induction variable.
  rewriter.replaceUsesWithIf(parentLoopOp.getInductionVar(),
                             parentLoopOp.getLowerBound(), [&](OpOperand &use) {
                               return ::llvm::is_contained(clonedOps,
                                                           use.getOwner());
                             });
  auto newDescOp = cast<xegpu::CreateNdDescOp>(clonedOps.back());

  // Compute offset for update operation: original offset - constant offset.
  llvm::SmallVector<Value> origDynamicOffsets, constDynamicOffsets,
      dynamicOffsets;
  llvm::SmallVector<int64_t> origStaticOffsets, constStaticOffsets,
      staticOffsets;
  dispatchIndexOpFoldResults(descOp.getMixedOffsets(), origDynamicOffsets,
                             origStaticOffsets);
  dispatchIndexOpFoldResults(newDescOp.getMixedOffsets(), constDynamicOffsets,
                             constStaticOffsets);
  int64_t dynIndex = 0;
  for (auto [i, origStaticOffset] : llvm::enumerate(origStaticOffsets)) {
    if (origStaticOffset != ShapedType::kDynamic) {
      // Original offset was a constant, difference must be 0.
      staticOffsets.push_back(0);
    } else {
      auto origDynOffset = origDynamicOffsets[dynIndex];
      auto cstDynOffset = constDynamicOffsets[dynIndex];
      auto subValue = rewriter.createOrFold<arith::SubIOp>(
          loc, origDynOffset.getType(), origDynOffset, cstDynOffset);
      auto maybeIntValue = getConstantIntValue(subValue);
      if (maybeIntValue) {
        // Folded to a constant int.
        staticOffsets.push_back(*maybeIntValue);
      } else {
        // Dynamic offset.
        dynamicOffsets.push_back(subValue);
        staticOffsets.push_back(ShapedType::kDynamic);
      }
      dynIndex++;
    }
  }

  // Insert an offset update op if non-trivial offset.
  bool allZeros = llvm::all_of(staticOffsets, [](int64_t s) { return s == 0; });
  if (!dynamicOffsets.empty() || !allZeros) {
    auto tile = newDescOp.getResult();
    auto offsetOp = rewriter.create<xegpu::UpdateNdOffsetOp>(
        loc, tile.getType(), tile, dynamicOffsets, staticOffsets);
    // replace subsequent uses of the descriptor with the offset descriptor
    rewriter.replaceUsesWithIf(
        descOp.getResult(), offsetOp.getResult(), [&](OpOperand &use) {
          return use.getOwner() != offsetOp.getOperation();
        });
  }
  rewriter.replaceOp(descOp, newDescOp);
  return newDescOp;
}

/// Add offset update ops after create desc ops in the loop body.
LogicalResult insertOffsetUpdateOps(transform::TransformRewriter &rewriter,
                                    scf::ForOp loopOp) {
  // Find all create desc operations in the loop body
  SmallVector<Operation *> createDescOps;
  for (auto &op : loopOp.getBody()->getOperations()) {
    if (isa<xegpu::CreateNdDescOp>(op)) {
      createDescOps.push_back(&op);
    }
  }
  if (createDescOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No xegpu.create_nd_desc ops found in the loop body.\n");
    return failure();
  }
  // Split to desc and offset update ops.
  for (auto &op : createDescOps) {
    auto descOp = cast<xegpu::CreateNdDescOp>(op);
    insertUpdateOp(rewriter, loopOp, descOp);
  }
  return success();
}

/// Check if an op can be hoisted out of the loop.
static bool canBeHoisted(Operation *op, LoopLikeOpInterface &loopLike) {
  return llvm::all_of(op->getOperands(), [&](Value value) {
    return loopLike.isDefinedOutsideOfLoop(value);
  });
}

/// Hoist create desc ops out of the loop.
/// If offset update ops exist, add values to loop iter_args and yield
FailureOr<scf::ForOp> hoistDescOps(transform::TransformRewriter &rewriter,
                                   scf::ForOp loopOp) {
  SmallVector<xegpu::CreateNdDescOp> descOps;
  auto loopLike = cast<LoopLikeOpInterface>(loopOp.getOperation());
  for (auto &op : loopOp.getBody()->getOperations()) {
    if (auto descOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
      if (canBeHoisted(descOp.getOperation(), loopLike)) {
        descOps.push_back(descOp);
      }
    }
  }
  if (descOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No hoistable create_nd_desc ops found in the loop body.\n");
    return loopOp;
  }

  SmallVector<Value> initValues, yieldValues;
  for (auto &descOp : descOps) {
    // We assume tensor desc is used by an offset update op, find it.
    auto maybeOffsetOp = getUserOfType<xegpu::UpdateNdOffsetOp>(descOp.getResult());
    if (!maybeOffsetOp) {
      continue;
    }
    auto offsetOp = *maybeOffsetOp;

    // Hoist desc op.
    auto producers =
        getProducerOpsInRegion(descOp.getOperation(), loopOp.getRegion(), true);
    for (auto &op : llvm::reverse(producers)) {
      rewriter.moveOpBefore(op, loopOp);
    }

    // Offset update op must be converted to increment the offset, instead of
    // defining an absolute offset wrt the original descriptor tile.
    // In offset update producer ops, replace loop variable with step size.
    auto offsetProducerOps =
        getProducerOpsInRegion(offsetOp.getOperation(), loopOp.getRegion());
    rewriter.replaceUsesWithIf(
        loopOp.getInductionVar(), loopOp.getStep(), [&](OpOperand &use) {
          return llvm::is_contained(offsetProducerOps, use.getOwner());
        });
    // Offsetted desc now points to next tile, users must use the current tile
    rewriter.replaceAllUsesWith(offsetOp.getResult(), offsetOp.getTensorDesc());
    // Add to loop init/yield values.
    initValues.push_back(descOp.getResult());
    yieldValues.push_back(offsetOp.getResult());
  }
  // Rewrite loop with new init/yield values.
  NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                 llvm::ArrayRef<BlockArgument> newBBArgs) {
    return yieldValues;
  };
  auto maybeNewLoop = loopOp.replaceWithAdditionalYields(
      rewriter, initValues,
      /*replaceInitOperandUsesInLoop=*/true, yieldFn);
  if (failed(maybeNewLoop)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to generate a new loop.\n");
    return failure();
  }
  return cast<scf::ForOp>(*maybeNewLoop);
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
  auto ctx = rewriter.getContext();
  auto oldTensorDesc = descOp.getResult();
  auto descShapedType = cast<ShapedType>(oldTensorDesc.getType());
  // This discards any block_tdesc_attr attributes.
  auto descType = xegpu::TensorDescType::get(ctx, descShapedType.getShape(),
                                             descShapedType.getElementType(),
                                             /*encoding=*/nullptr,
                                             /*layout=*/layout);

  rewriter.setInsertionPointAfter(descOp);
  auto newDescOp = rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descType, descOp.getSource(), descOp.getMixedOffsets(),
      descOp.getMixedSizes(), descOp.getMixedStrides());

  return newDescOp;
}

/// Fuse two scf.for loops into one. Keeps track of source operations to their
/// cloned targets. Returns the new fused loop.
scf::ForOp fuseForLoops(scf::ForOp target, scf::ForOp source,
                        RewriterBase &rewriter,
                        SmallVector<Operation *> &sourceOps,
                        SmallVector<Operation *> &targetOps) {
  // This method is modified from mlir::fuseIndependentSiblingForLoops to
  // trace the source ops to their cloned targets.

  unsigned numTargetOuts = target.getNumResults();
  unsigned numSourceOuts = source.getNumResults();

  // Create fused init_args, with target's init_args before source's init_args.
  SmallVector<Value> fusedInitArgs;
  llvm::append_range(fusedInitArgs, target.getInitArgs());
  llvm::append_range(fusedInitArgs, source.getInitArgs());

  // Create a new scf.for op after the source loop (with scf.yield terminator
  // (without arguments) only in case its init_args is empty).
  rewriter.setInsertionPointAfter(source);
  scf::ForOp fusedLoop = rewriter.create<scf::ForOp>(
      source.getLoc(), source.getLowerBound(), source.getUpperBound(),
      source.getStep(), fusedInitArgs);

  // Map original induction variables and operands to those of the fused loop.
  IRMapping mapping;
  mapping.map(target.getInductionVar(), fusedLoop.getInductionVar());
  mapping.map(target.getRegionIterArgs(),
              fusedLoop.getRegionIterArgs().take_front(numTargetOuts));
  mapping.map(source.getInductionVar(), fusedLoop.getInductionVar());
  mapping.map(source.getRegionIterArgs(),
              fusedLoop.getRegionIterArgs().take_back(numSourceOuts));

  // Merge target's body into the new (fused) for loop and then source's body.
  rewriter.setInsertionPointToStart(fusedLoop.getBody());
  IRMapping clonedOpsMapping;
  for (Operation &op : target.getBody()->without_terminator()) {
    auto newOp = rewriter.clone(op, mapping);
    clonedOpsMapping.map(&op, newOp);
  }
  for (Operation &op : source.getBody()->without_terminator()) {
    auto newOp = rewriter.clone(op, mapping);
    clonedOpsMapping.map(&op, newOp);
  }
  // Map the given source operations to their cloned targets.
  auto opsMap = clonedOpsMapping.getOperationMap();
  for (Operation *op : sourceOps) {
    auto it = opsMap.find(op);
    if (it != opsMap.end()) {
      targetOps.push_back(it->second);
    } else {
      targetOps.push_back(nullptr);
    }
  }

  // Build fused yield results by appropriately mapping original yield operands.
  SmallVector<Value> yieldResults;
  for (Value operand : target.getBody()->getTerminator()->getOperands())
    yieldResults.push_back(mapping.lookupOrDefault(operand));
  for (Value operand : source.getBody()->getTerminator()->getOperands())
    yieldResults.push_back(mapping.lookupOrDefault(operand));
  if (!yieldResults.empty())
    rewriter.create<scf::YieldOp>(source.getLoc(), yieldResults);

  // Replace old loops by substituting their uses by results of the fused loop.
  rewriter.replaceOp(target, fusedLoop.getResults().take_front(numTargetOuts));
  rewriter.replaceOp(source, fusedLoop.getResults().take_back(numSourceOuts));

  return fusedLoop;
}

DiagnosedSilenceableFailure transform::HoistDescOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto loopOp = dyn_cast<scf::ForOp>(target);
  if (!loopOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a scf.for op, but got: " << target->getName();
  }

  if (failed(insertOffsetUpdateOps(rewriter, loopOp))) {
    return emitSilenceableFailure(getLoc())
           << "No desc ops found in the loop body " << target->getName();
  }
  auto newLoopOp = hoistDescOps(rewriter, loopOp);
  if (failed(newLoopOp)) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Failed to hoist xegpu.create_nd_desc ops";
    diag.attachNote(loopOp.getLoc()) << "loop op";
    return diag;
  }
  loopOp = *newLoopOp;
  results.push_back(loopOp.getOperation());
  return DiagnosedSilenceableFailure::success();
}

void transform::HoistDescOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLoopMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::InsertPrefetchOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {

  auto dpasOps = state.getPayloadOps(getDpasOp());
  auto loopOps = state.getPayloadOps(getLoopOp());

  if (!llvm::hasSingleElement(dpasOps)) {
    return emitDefiniteFailure() << "requires exactly one dpasOp handle (got "
                                 << llvm::range_size(dpasOps) << ")";
  }
  if (!llvm::hasSingleElement(loopOps)) {
    return emitDefiniteFailure() << "requires exactly one loopOp handle (got "
                                 << llvm::range_size(loopOps) << ")";
  }

  Operation *dpasPtr = *dpasOps.begin();
  auto dpasOp = dyn_cast<xegpu::DpasOp>(dpasPtr);
  if (!dpasOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a xegpu.dpas op, but got: " << dpasPtr->getName();
  }

  Operation *loopPtr = *loopOps.begin();
  auto forOp = dyn_cast<scf::ForOp>(loopPtr);
  if (!forOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected a scf.for op, but got: " << loopPtr->getName();
  }

  auto parentLoop = dpasOp->getParentOfType<scf::ForOp>();
  if (!parentLoop || parentLoop != forOp) {
    return emitSilenceableFailure(getLoc())
           << "dpasOp is not contained in the given scf.for loop.";
  }

  int64_t tileIndex = getTileIndex();
  if (tileIndex >= dpasOp.getNumOperands()) {
    return emitSilenceableFailure(getLoc())
           << "tileIndex exceeds the number of op operands.";
  }

  auto sgLayout = getSgLayout();
  if (sgLayout.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_layout to be a 2D vector";
  }

  auto sgData = getSgData();
  if (sgData.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_data to be a 2D vector";
  }

  // Find descriptor op of the operand.
  Value opVec = dpasOp.getOperation()->getOperand(tileIndex);
  auto maybeDescOp = findDescriptorOp(opVec, dpasOp.getOperation());
  if (!maybeDescOp) {
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  }
  auto descOp = *maybeDescOp;

  // Clone reduction loop.
  rewriter.setInsertionPoint(forOp);
  auto newForOp =
      rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                  forOp.getUpperBound(), forOp.getStep());
  // Clone desc op into it.
  rewriter.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  auto newDescOp = cast<xegpu::CreateNdDescOp>(
      rewriter.clone(*descOp.getOperation(), mapping));
  // Set desc op layout.
  auto layout = createLayoutAttr(rewriter.getContext(), sgLayout, sgData,
                                 /*instData=*/std::nullopt);
  newDescOp = setDescLayout(rewriter, newDescOp, layout);

  // Insert prefetch op.
  auto ctx = rewriter.getContext();
  auto readCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);
  rewriter.create<xegpu::PrefetchNdOp>(newDescOp.getLoc(),
                                       newDescOp.getResult(), readCacheHint,
                                       readCacheHint, readCacheHint);

  // Insert offset update op.
  insertUpdateOp(rewriter, newForOp, newDescOp);
  // Hoist descriptor op out of the loop.
  auto maybenewForOp = hoistDescOps(rewriter, newForOp);
  if (failed(maybenewForOp)) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Failed to hoist xegpu.create_nd_desc ops";
    diag.attachNote(newForOp.getLoc()) << "loop op";
    return diag;
  }
  newForOp = *maybenewForOp;

  // Peel first iteration of the loop and reset lower bound to original value.
  scf::ForOp firstLoopOp;
  if (failed(scf::peelForLoopFirstIteration(rewriter, newForOp, firstLoopOp))) {
    auto diag = emitSilenceableFailure(getLoc()) << "Failed to peel the loop";
  }
  newForOp.setLowerBound(forOp.getLowerBound());

  // Fuse with the original loop, keep track of cloned ops.
  SmallVector<Operation *> sourceOps{dpasOp.getOperation()}, targetOps;
  auto fusedLoop =
      fuseForLoops(newForOp, forOp, rewriter, sourceOps, targetOps);
  assert(fusedLoop && "failed to fuse loops");

  // Get the cloned dpas op.
  auto clonedDpasOp = targetOps[0];
  if (!clonedDpasOp) {
    return emitSilenceableFailure(getLoc())
           << "Failed to find cloned dpas op in the fused loop.";
  }

  // Map result handles.
  results.set(cast<OpResult>(getTransformedLoopOp()), {fusedLoop});
  results.set(cast<OpResult>(getTransformedDpasOp()), {clonedDpasOp});

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::SetDPASLayoutOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto dpasOp = dyn_cast<xegpu::DpasOp>(target);
  if (!dpasOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a xegpu.dpas op, but got: " << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  int64_t tileIndex = getTileIndex();
  if (tileIndex >= dpasOp.getNumOperands()) {
    return emitSilenceableFailure(getLoc())
           << "tileIndex exceeds the number of op operands.";
  }

  auto sgLayout = getSgLayout();
  if (sgLayout.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_layout to be a 2D vector";
  }

  auto sgData = getSgData();
  if (sgData.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected sg_data to be a 2D vector";
  }

  auto instData = getInstData();
  if (instData.size() != 2) {
    return emitSilenceableFailure(getLoc())
           << "Expected inst_data to be a 2D vector";
  }

  llvm::ArrayRef<int> loadData = instData;
  if (getLoadData().has_value()) {
    loadData = getLoadData().value();
    if (loadData.size() != 2) {
      return emitSilenceableFailure(getLoc())
            << "Expected load_data to be a 2D vector";
    }
    if (loadData[0] < instData[0] || loadData[1] < instData[1]) {
      return emitSilenceableFailure(getLoc())
      << "load_data size must be larger or equal to inst_data size";
    }
    if (loadData[0] % instData[0] != 0 || loadData[1] % instData[1] != 0) {
      return emitSilenceableFailure(getLoc())
      << "load_data must be evenly divisible by inst_data";
    }
  }

  // Replace descriptor op using layout attribute.
  Value opVec = dpasOp.getOperation()->getOperand(tileIndex);
  auto maybeDescOp = findDescriptorOp(opVec, dpasOp.getOperation());
  if (!maybeDescOp) {
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  }
  auto descOp = *maybeDescOp;
  // Layout for the load op.
  auto loadLayoutAttr = createLayoutAttr(rewriter.getContext(), sgLayout, sgData, loadData);
  descOp = setDescLayout(rewriter, descOp, loadLayoutAttr);
  // Layout for the instruction.
  auto instLayoutAttr = createLayoutAttr(rewriter.getContext(), sgLayout, sgData, instData);
  if (tileIndex == 2) {
    // C operand: set layout attribute for the dpas op result
    xegpu::setLayoutAttr(dpasOp.getOperation()->getResults()[0], instLayoutAttr);
  }

  if (loadLayoutAttr != instLayoutAttr) {
    // Insert convert layout op after load op.
    auto maybeLoadOp = getUserOfType<xegpu::LoadNdOp>(descOp.getResult());
    if (!maybeLoadOp) {
      return emitSilenceableFailure(getLoc())
      << "Expected a xegpu.load_nd op as a user of the descriptor op."; 
    }
    auto loadOp = *maybeLoadOp;
    rewriter.setInsertionPointAfter(loadOp.getOperation());
    auto source = loadOp.getResult();
    auto convLayoutOp = rewriter.create<xegpu::ConvertLayoutOp>(
        loadOp.getLoc(), source.getType(), source,
        loadLayoutAttr, instLayoutAttr);
    // Replace load op result with the converted layout.
    rewriter.replaceUsesWithIf(
      source, convLayoutOp.getResult(),
      [&](OpOperand &use) {
        return use.getOwner() != convLayoutOp.getOperation();
      });
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::SetDPASLayoutOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getDpasOpMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform::SetGPULaunchThreadsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  auto launchOp = dyn_cast<gpu::LaunchOp>(target);
  if (!launchOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a gpu.launch op, but got: " << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  auto threads = getThreads();
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
  onlyReadsHandle(getLaunchOpMutable(), effects);
  modifiesPayload(effects);
}
