// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @hoist_desc_ops
func.func @hoist_desc_ops(%arg0: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-NEXT: scf.for
  scf.for %arg1 = %c0 to %c4096 step %c32 {
    // CHECK: xegpu.update_nd_offset
    // CHECK: xegpu.load_nd
    %0 = xegpu.create_nd_tdesc %arg0[0, %arg1] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
    %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: = transform.xegpu.hoist_desc_ops %{{.*}}
    %1 = transform.xegpu.hoist_desc_ops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @get_desc_op
func.func @get_desc_op(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: %[[V1:.+]] = xegpu.load_nd %[[V0]]
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas %[[V1]]
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.get_desc_op %{{.*}}
    %1 = transform.xegpu.get_desc_op %0 index = 0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_operand_layout_a
func.func @set_operand_layout_a(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>>
  %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: %[[V1:.+]] = xegpu.load_nd %[[V0]]
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas %[[V1]]
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_operand_layout %{{.*}}
    transform.xegpu.set_operand_layout %0 index = 0 sg_layout = [8, 4] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_operand_layout_b
func.func @set_operand_layout_b(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  // CHECK: = xegpu.create_nd_tdesc
  %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: = xegpu.load_nd
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg1
  %2 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  // CHECK-SAME: #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>>
  // CHECK: %[[V1:.+]] = xegpu.load_nd %[[V0]]
  %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas %1, %[[V1]]
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_operand_layout %{{.*}}
    transform.xegpu.set_operand_layout %0 index = 1 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [16, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_operand_layout_c
func.func @set_operand_layout_c(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  // CHECK: = xegpu.create_nd_tdesc
  %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: = xegpu.load_nd
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: = xegpu.create_nd_tdesc
  %2 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  // CHECK: = xegpu.load_nd
  %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg2
  %4 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  // CHECK-SAME: #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>>
  // CHECK: %[[V1:.+]] = xegpu.load_nd %[[V0]]
  %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas %1, %3, %[[V1]] {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_operand_layout %{{.*}}
    transform.xegpu.set_operand_layout %0 index = 2 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @insert_prefetch_dpas_a
func.func @insert_prefetch_dpas_a(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: !xegpu.tensor_desc<256x32xf16, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32]>>
  // Peeled first iteration of the loop, canonicalization drops the scf.for
  // CHECK: %[[V2:.+]] = scf.for
  // CHECK-SAME: iter_args(%[[V1:.+]] = %[[V0]])
  // CHECK: = xegpu.update_nd_offset %[[V1]], [0, %[[C32]]]
  // CHECK: xegpu.prefetch_nd %[[V1]]
  // Reduction loop
  // CHECK: scf.for
  // CHECK-SAME: iter_args(%[[V3:.+]] = %[[V2]]
  %2 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %1) -> (vector<256x256xf16>) {
    // CHECK: = xegpu.update_nd_offset %[[V3]], [0, %[[C32]]]
    // CHECK: xegpu.prefetch_nd %[[V3]]
    %3 = xegpu.create_nd_tdesc %arg0[0, %arg3] : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
    %4 = xegpu.load_nd %3  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
    %5 = xegpu.create_nd_tdesc %arg1[%arg3, 0] : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
    %6 = xegpu.load_nd %5  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
    %7 = xegpu.dpas %4, %6, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
    scf.yield %7 : vector<256x256xf16>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["xegpu.dpas"]} in %0 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.insert_prefetch %{{.*}} %{{.*}}
    %2, %3 = transform.xegpu.insert_prefetch %1 %0 index = 0 sg_layout = [32, 1] sg_data = [8, 32] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_gpu_launch_threads
func.func @set_gpu_launch_threads(%arg0: memref<4096x4096xf16>) {
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C16:.+]] = arith.constant 16 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1_0:.+]] = arith.constant 1 : index
  // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C16]], %{{.*}} = %[[C16]], %{{.*}} = %[[C1]])
  // CHECK-SAME: threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C8]], %{{.*}} = %[[C4]], %{{.*}} = %[[C1_0]])
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c16, %arg10 = %c16, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    gpu.terminator
  } {SCFToGPU_visited}
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["gpu.launch"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_gpu_launch_threads %{{.*}}
    transform.xegpu.set_gpu_launch_threads %0 threads = [8, 4, 1] : !transform.any_op
    transform.yield
  }
}
