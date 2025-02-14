// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   -buffer-deallocation-simplification -split-input-file %s | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose() -> memref<2x3xf32> {
  %alloc0 = memref.alloc() : memref<3x2xf32>
  %alloc1 = memref.alloc() : memref<2x3xf32>
  %0 = memref.transpose %alloc0 (d0, d1) -> (d1, d0) : memref<3x2xf32> to memref<2x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 2)>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<2x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 2)>>) outs(%alloc1 : memref<2x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return %alloc1 : memref<2x3xf32>
}
// CHECK-LABEL: func @transpose
