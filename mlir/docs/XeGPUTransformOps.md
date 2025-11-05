# RFC for XeGPU Transform ops

## Summary

The XeGPU dialect capabilities need to be extended upstream to facilitate more generic lowering of high-level operations. Specifically we need to be able to lower `linalg` dialect operations to XeGPU dialect.

Existing upstream transform dialect operations can be used to prepare a matmul operation for XeGPU lowering, for example, by applying appropriate tiling transforms, but many XeGPU specific transform patterns are still missing.

This RFC outlines the following new operations to fill the gaps. These operators reside in the XeGPU namespace of the transform dialect:

* `transform.xegpu.get_desc_op`: Find the defining `xegpu.create_nd_tdesc` operation of an operand.
* `transform.xegpu.set_desc_layout`: Attach `xegpu.layout` attribute to the descriptor that `xegpu.create_nd_tdesc` op returns.
* `transform.xegpu.convert_operand_layout`: Emit `xegpu.convert_layout` op to change the `xegpu.layout` of an operand.
* `transform.xegpu.set_op_layout_attr`: Set `xegpu.layout` attribute of an operation's result or operand.
* `transform.xegpu.insert_prefetch`: Inserts XeGPU cooperative prefetch operations to an op operand.
* `transform.xegpu.set_gpu_launch_threads`: Set number of threads for a given `gpu.launch` operation.

These operations are sufficient for lowering a `linalg.matmul` operation to XeGPU dialect and provide necessary transforms to obtain good performance on Intel GPUs (PVC/BMG, 4k matmul benchmark).

## Example: 4k matrix multiplication payload

Consider the following 4k `linalg.matmul` payload function.

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>)
    attributes {llvm.emit_c_interface} {
  %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
  %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
  %3 = linalg.matmul ins(%0, %1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%2 : tensor<4096x4096xf32>) ->
    tensor<4096x4096xf32>
  bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
  return
}
```

Note that the matmul is defined with `tensor` operands because in general we need tensor semantics to perform certain optimizations (loop fusion, loop-invariant code motion).

First, we tile the payload to GPU work groups (WG) and K-tiles using the following upstream transform operations:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.matmul"]} in %0 : (!transform.any_op) -> !transform.any_op
    // WG tiling
    %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [256, 256] : (!transform.any_op) ->
      (!transform.any_op, !transform.any_op)
    // tile k-dimension
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_op tile_sizes [0, 0, 32] : (!transform.any_op) ->
      (!transform.any_op, !transform.any_op)
    transform.apply_cse to %0 : !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    ...
    transform.yield
  }
}
```

After the transformations the payload function is correctly tiled to `scf.forall` WG loop, followed by a sequential `scf.for` reduction loop.

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>)
    attributes {llvm.emit_c_interface} {
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
  %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
  %3 = scf.forall (%arg3, %arg4) in (16, 16) shared_outs(%arg5 = %2) -> (tensor<4096x4096xf32>) {
    %4 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg3)
    %5 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg4)
    %extracted_slice = tensor.extract_slice %0[%4, 0] [256, 4096] [1, 1] : tensor<4096x4096xf16> to tensor<256x4096xf16>
    %extracted_slice_0 = tensor.extract_slice %1[0, %5] [4096, 256] [1, 1] : tensor<4096x4096xf16> to tensor<4096x256xf16>
    %extracted_slice_1 = tensor.extract_slice %arg5[%4, %5] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
    %6 = scf.for %arg6 = %c0 to %c4096 step %c32 iter_args(%arg7 = %extracted_slice_1) -> (tensor<256x256xf32>) {
      %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg6] [256, 32] [1, 1] : tensor<256x4096xf16> to
        tensor<256x32xf16>
      %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [32, 256] [1, 1] : tensor<4096x256xf16> to
        tensor<32x256xf16>
      %7 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<256x32xf16>, tensor<32x256xf16>)
        outs(%arg7 : tensor<256x256xf32>) -> tensor<256x256xf32>
      scf.yield %7 : tensor<256x256xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %6 into %arg5[%4, %5] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<4096x4096xf32>
    }
  }
  bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
  return
}
```

We can now lower the `linalg.matmul` operations to the vector dialect and hoist loop-invariant reads and writes out of the reduction loop:

```mlir
    %2 = transform.structured.vectorize_children_and_apply_patterns %0 {fold_type_extensions_into_contract} :
      (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["scf.for"]} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %3 : !transform.any_op
    transform.apply_cse to %2 : !transform.any_op
    transform.apply_patterns to %2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
```

resulting in:

```mlir
  ...
  %0 = ub.poison : f32
  %1 = ub.poison : f16
  %5 = scf.forall (%arg3, %arg4) in (16, 16) shared_outs(%arg5 = %4) -> (tensor<4096x4096xf32>) {
    %6 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg3)
    %7 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg4)
    %extracted_slice = tensor.extract_slice %arg5[%6, %7] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
    %8 = vector.transfer_read %extracted_slice[%c0, %c0], %0 {in_bounds = [true, true]} :
      tensor<256x256xf32>, vector<256x256xf32>
    %9 = scf.for %arg6 = %c0 to %c4096 step %c32 iter_args(%arg7 = %8) -> (vector<256x256xf32>) {
      %11 = vector.transfer_read %2[%6, %arg6], %1 {in_bounds = [true, true]} : tensor<4096x4096xf16>, vector<256x32xf16>
      %12 = vector.transfer_read %3[%arg6, %7], %1 {in_bounds = [true, true]} : tensor<4096x4096xf16>, vector<32x256xf16>
      %13 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
        %11, %12, %arg7 : vector<256x32xf16>, vector<32x256xf16> into vector<256x256xf32>
      scf.yield %13 : vector<256x256xf32>
    }
    %10 = vector.transfer_write %9, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>,
      tensor<256x256xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg5[%6, %7] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<4096x4096xf32>
    }
  }
  ...
```

Next we bufferize the function and convert the parallel loop to a `gpu.launch` op,

```mlir
    // bufferize
    %4 = transform.get_parent_op %2 {deduplicate, op_name = "builtin.module"} : (!transform.any_op) -> !transform.any_op
    %5 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %4 {allow_return_allocs_from_loops = true,
      bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op
    %6 = transform.apply_registered_pass "fold-memref-alias-ops" to %5 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %6 : !transform.any_op
    transform.apply_patterns to %6 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    // map to gpu.launch
    %7 = transform.structured.match ops{["scf.forall"]} in %6 : (!transform.any_op) -> !transform.any_op
    %8 = transform.loop.forall_to_parallel %7 : (!transform.any_op) -> !transform.any_op
    %9 = transform.get_parent_op %8 : (!transform.any_op) -> !transform.any_op
    %10 = transform.apply_registered_pass "gpu-map-parallel-loops" to %9 : (!transform.any_op) -> !transform.any_op
    %11 = transform.apply_registered_pass "convert-parallel-loops-to-gpu" to %10 : (!transform.any_op) -> !transform.any_op
    %12 = transform.apply_registered_pass "lower-affine" to %11 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %12 : !transform.any_op
    transform.apply_patterns to %12 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
```

resulting in:

```mlir
  func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>)
      attributes {llvm.emit_c_interface} {
    %c256 = arith.constant 256 : index
    %0 = ub.poison : f32
    %1 = ub.poison : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c16, %arg10 = %c16, %arg11 = %c1) threads(%arg6, %arg7, %arg8)
        in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      %2 = arith.muli %arg3, %c256 overflow<nsw> : index
      %3 = arith.muli %arg4, %c256 overflow<nsw> : index
      %4 = vector.transfer_read %arg2[%2, %3], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<256x256xf32>
      %5 = scf.for %arg15 = %c0 to %c4096 step %c32 iter_args(%arg16 = %4) -> (vector<256x256xf32>) {
        %6 = vector.transfer_read %arg0[%2, %arg15], %1 {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<256x32xf16>
        %7 = vector.transfer_read %arg1[%arg15, %3], %1 {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<32x256xf16>
        %8 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"],
          kind = #vector.kind<add>} %6, %7, %arg16 : vector<256x32xf16>, vector<32x256xf16> into vector<256x256xf32>
        scf.yield %8 : vector<256x256xf32>
      }
      vector.transfer_write %5, %arg2[%2, %3] {in_bounds = [true, true]} : vector<256x256xf32>, memref<4096x4096xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
```

### `xegpu.set_gpu_launch_threads` operation

Above, the `gpu.launch` op correctly uses (16, 16, 1) blocks but has only a single thread (1, 1, 1). Because the subgroup (SG) distribution will be handled later on by XeGPU distribution pass, we need to set the correct number of threads manually:

```mlir
    %13 = transform.structured.match ops{["gpu.launch"]} in %12 : (!transform.any_op) -> !transform.any_op
    transform.xegpu.set_gpu_launch_threads %13 threads = [1024, 1, 1] : !transform.any_op
    transform.apply_patterns to %12 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
```

after which the launch op becomes:

```mlir
  ...
  %c1024 = arith.constant 1024 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c16, %arg10 = %c16, %arg11 = %c1) threads(%arg6, %arg7, %arg8)
    in (%arg12 = %c1024, %arg13 = %c1, %arg14 = %c1) {
    ...
```

### Conversion to XeGPU dialect

We can now outline the GPU kernel and convert vector ops to xegpu dialect using the `convert-vector-to-xegpu` pass:

```mlir
    // kernel outlining
    %14 = transform.apply_registered_pass "gpu-launch-sink-index-computations" to %12 : (!transform.any_op) ->
      !transform.any_op
    %15 = transform.apply_registered_pass "gpu-kernel-outlining" to %6 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %15 : !transform.any_op
    // convert vector to xegpu
    %16 = transform.apply_registered_pass "xevm-attach-target" with options = {"O" = "3", "chip" = "bmg"} to %15 :
      (!transform.any_op) -> !transform.any_op
    %17 = transform.structured.match ops{["gpu.module"]} in %16 : (!transform.any_op) -> !transform.any_op
    %18 = transform.structured.match ops{["gpu.func"]} in %17 : (!transform.any_op) -> !transform.any_op
    %19 = transform.apply_registered_pass "convert-vector-to-xegpu" to %18 : (!transform.any_op) -> !transform.any_op
```

The payload function now reads:

```mlir
func.func @run(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>)
    attributes {llvm.emit_c_interface} {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  gpu.launch_func  @run_kernel::@run_kernel blocks in (%c16, %c16, %c1) threads in (%c1024, %c1, %c1)
    args(%arg2 : memref<4096x4096xf32>, %arg0 : memref<4096x4096xf16>, %arg1 : memref<4096x4096xf16>)
  return
}
gpu.module @run_kernel [#xevm.target<O = 3>] {
  gpu.func @run_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>)
      kernel attributes {known_block_size = array<i32: 1024, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
    %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
    %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32>
    %3 = xegpu.load_nd %2[%0, %1]  : !xegpu.tensor_desc<256x256xf32> -> vector<256x256xf32>
    %4 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %3) -> (vector<256x256xf32>) {
      %6 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
      %7 = xegpu.load_nd %6[%0, %arg3]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
      %8 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
      %9 = xegpu.load_nd %8[%arg3, %1]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
      %10 = xegpu.dpas %7, %9, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
      scf.yield %10 : vector<256x256xf32>
    }
    %5 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32>
    xegpu.store_nd %4, %5[%0, %1]  : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32>
    gpu.return
  }
}
```

The payload IR has now been converted to XeGPU dialect at WG level. It however lacks the `xegpu.layout` attributes to define how the ops should be mapped to SG and instruction level.

### Attaching `xegpu.layout` attributes

#### `xegpu.get_desc_op` and `xegpu.set_desc_layout` operations

We start by adding layouts to the DPAS op's A, B, and C operands.

We use the `xegpu.get_desc_op` transform op to find the defining `xegpu.create_nd_tdesc` op of the operands. The transform op takes a handle to the DPAS op and the desired operand defined with the `index` argument. For the A operand, we use:

```mlir
    // match the dpas op in scf.for
    %k_loop = transform.structured.match ops{["scf.for"]} in %19 : (!transform.any_op) -> !transform.any_op
    %dpas_op = transform.structured.match ops{["xegpu.dpas"]} in %k_loop : (!transform.any_op) -> !transform.any_op
    // find desc op for tile A (index = 0)
    %desc_op = transform.xegpu.get_desc_op %dpas_op index = 0 : (!transform.any_op) -> !transform.any_op
```

Once we have a handle to the desc op, we can set the desc layout with the `xegpu.set_desc_layout` transform op.

```mlir
    %new_desc_op = transform.xegpu.set_desc_layout %desc_op sg_layout = [8, 8] sg_data = [32, 32] inst_data = [32, 16] :
      (!transform.any_op) -> !transform.any_op
```

Because this transform alters the return value of the `xegpu.create_nd_tdesc` op, the op is replaced with a new one. The transform op returns a handle to the new desc op and the old handle (`%desc_op`) is invalidated. After applying the transform, the A descriptor reads:

```mlir
        ...
        %5 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16,
          #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>>
        ...
```

The B and C operands are handled analogously.

#### `xegpu.convert_operand_layout` operation

Above we use `inst_data = [32, 16]` for the A tile layout. This value is too large for the DPAS op. We use ``xegpu.convert_operand_layout` transform op to emit a `xegpu.convert_layout` op to change `inst_data` to the expected `[8, 16]` value. The op again takes a handle to the DPAS op and an index defining the operand:

```mlir
    transform.xegpu.convert_operand_layout %dpas_op index = 0 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] :
      !transform.any_op
```

The payload IR now reads:

```mlir
        %5 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16,
          #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>>
        %6 = xegpu.load_nd %5[%0, %arg3]  : !xegpu.tensor_desc<256x32xf16,
          #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>> -> vector<256x32xf16>
        %7 = xegpu.convert_layout %6 <{
          input_layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>,
          target_layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<256x32xf16>
        ...
        %10 = xegpu.dpas %7, %9, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
```

That is, we have defined the layout for the A operand, and configured it to use a larger `[32, 16]` tile size for loading the data. The layout conversion causes the subsequent xegpu lowering passes to unroll the larger tile to the expected DPAS instruction size `[8, 16]`.

B tile layout conversion is configured analogously.

#### `xegpu.set_op_layout_attr` operation

Above we have set the layout attribute to the DPAS op's operands `xegpu.tensor_desc` descriptors. We also need to set the layout attribute of some ops, for example, the DPAS op (which operates on `vector`s) in this case. To this end we can use the `xegpu.set_op_layout_attr` transform op. Similarly to the other ops, it takes a handle to the the payload op and an index to define the operand (default value is 0). It also has an optional `result` argument which sets op result attribute instead.

The following transform op annotates the DPAS op with the C layout for the result value:

```mlir
    transform.xegpu.set_op_layout_attr %dpas_op result sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] :
      !transform.any_op
```

which sets the `layout_result_0` attribute to the DPAS op:

```mlir
        %11 = xegpu.dpas %7, %10, %arg4 {layout_result_0 = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32],
          inst_data = [8, 16]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
```

If the `result` argument were left out, the transform would set the `layout_operand_0` attribute.

The `xegpu.set_op_layout_attr` transform op is also needed to annotate other intermediate ops that operate on `vector`s, such as pre/post elementwise ops (e.g. `arith.addf`) or type conversions (e.g., `arith.extf`).

### Adding cooperative prefetching

The only missing ingredient is emitting prefetch op for the matmul tiles. The `o` transform op takes a handle to the `scf.for` reduction loop and the `xegpu.dpas` op. Again, by specifying the operand index we can emit prefetch ops for the A or B operands. The transfrom takes the same `xegpu.layout` arguments, in this case to be applied to the `xegpu.prefetch_nd` op. There's also a `nb_prefetch` argument specifying how many steps ahead one wants to prefetch.

The following transform op will emit 2-steps-ahead prefetch pattern for the A tile:

```mlir
transform.xegpu.insert_prefetch %dpas_op %k_loop index = 0 sg_layout = [32, 1] sg_data = [8, 32] inst_data = [8, 16]
  nb_prefetch = 2 : !transform.any_op, !transform.any_op
```

resulting in:

```mlir
      ...
      %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16,
        #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
      xegpu.prefetch_nd %4[%0, %c0] : !xegpu.tensor_desc<256x32xf16,
        #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
      xegpu.prefetch_nd %4[%0, %c32] : !xegpu.tensor_desc<256x32xf16,
        #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
      %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %3) -> (vector<256x256xf32>) {
        %6 = arith.addi %arg3, %c64 : index
        xegpu.prefetch_nd %4[%0, %6] : !xegpu.tensor_desc<256x32xf16,
          #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
        ...
        scf.yield %13 : vector<256x256xf32>
      }
      ...
```

### Full XeGPU workgroup level example

The following transform schedule sets all the necessary xegpu layout attributes for the matmul payload and inserts prefetches (1-step-ahead):

```mlir
    %20 = transform.structured.match ops{["scf.for"]} in %19 : (!transform.any_op) -> !transform.any_op
    %21 = transform.structured.match ops{["xegpu.dpas"]} in %20 : (!transform.any_op) -> !transform.any_op
    %22 = transform.xegpu.get_desc_op %21 : (!transform.any_op) -> !transform.any_op
    %23 = transform.xegpu.set_desc_layout %22 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [32, 16] :
      (!transform.any_op) -> !transform.any_op
    transform.xegpu.convert_operand_layout %21 index = 0 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] :
      !transform.any_op
    %24 = transform.xegpu.get_desc_op %21 index = 1 : (!transform.any_op) -> !transform.any_op
    %25 = transform.xegpu.set_desc_layout %24 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [32, 16] :
      (!transform.any_op) -> !transform.any_op
    transform.xegpu.convert_operand_layout %21 index = 1 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [16, 16] :
      !transform.any_op
    %26 = transform.xegpu.get_desc_op %21 index = 2 : (!transform.any_op) -> !transform.any_op
    %27 = transform.xegpu.set_desc_layout %26 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] :
      (!transform.any_op) -> !transform.any_op
    transform.xegpu.set_op_layout_attr %21 result sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] :
      !transform.any_op
    transform.xegpu.insert_prefetch %21 %20 index = 0 sg_layout = [32, 1] sg_data = [8, 32] inst_data = [8, 16] :
      !transform.any_op, !transform.any_op
    transform.xegpu.insert_prefetch %21 %20 index = 1 sg_layout = [4, 16] sg_data = [8, 16] inst_data = [8, 16] :
      !transform.any_op, !transform.any_op
    transform.apply_cse to %19 : !transform.any_op
    transform.apply_patterns to %19 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_licm to %20 : !transform.any_op
    transform.apply_patterns to %19 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %19 : !transform.any_op
```

Note that towards the end we are using `transform.apply_licm` to hoist the `xegpu.create_nd_desc` ops out of the reduction loop.

After applying the transforms, the gpu kernel becomes:

```mlir
gpu.module @run_kernel [#xevm.target<O = 3>] {
  gpu.func @run_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>)
      kernel attributes {known_block_size = array<i32: 1024, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
    %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
    %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32,
      #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>>
    %3 = xegpu.load_nd %2[%0, %1]  : !xegpu.tensor_desc<256x256xf32,
      #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>> -> vector<256x256xf32>
    %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16,
      #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
    xegpu.prefetch_nd %4[%0, %c0] : !xegpu.tensor_desc<256x32xf16,
      #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
    %5 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16,
      #xegpu.layout<sg_layout = [4, 16], sg_data = [8, 16], inst_data = [8, 16]>>
    xegpu.prefetch_nd %5[%c0, %1] : !xegpu.tensor_desc<32x256xf16,
      #xegpu.layout<sg_layout = [4, 16], sg_data = [8, 16], inst_data = [8, 16]>>
    %6 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16,
      #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>>
    %7 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16,
      #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>>
    %8 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %3) -> (vector<256x256xf32>) {
      %9 = arith.addi %arg3, %c32 : index
      xegpu.prefetch_nd %5[%9, %1] : !xegpu.tensor_desc<32x256xf16,
        #xegpu.layout<sg_layout = [4, 16], sg_data = [8, 16], inst_data = [8, 16]>>
      xegpu.prefetch_nd %4[%0, %9] : !xegpu.tensor_desc<256x32xf16,
        #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>>
      %10 = xegpu.load_nd %6[%0, %arg3]  : !xegpu.tensor_desc<256x32xf16,
        #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>> -> vector<256x32xf16>
      %11 = xegpu.convert_layout %10 <{input_layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32],
        inst_data = [32, 16]>, target_layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>}> :
        vector<256x32xf16>
      %12 = xegpu.load_nd %7[%arg3, %1]  : !xegpu.tensor_desc<32x256xf16,
        #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [32, 16]>> -> vector<32x256xf16>
      %13 = xegpu.convert_layout %12 <{input_layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32],
        inst_data = [32, 16]>, target_layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [16, 16]>}> :
        vector<32x256xf16>
      %14 = xegpu.dpas %11, %13, %arg4 {layout_result_0 = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32],
        inst_data = [8, 16]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
      scf.yield %14 : vector<256x256xf32>
    }
    xegpu.store_nd %8, %2[%0, %1]  : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32,
      #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>>
    gpu.return
  }
}
```

The above IR is now fully annotated and can be lowered to XeGPU SG and instruction level using `xegpu-wg-to-sg-distribute` and `xegpu-blocking` passes, respectively, and further to binary using the upstream XeVM pipeline.

### Tunable parameters

The above schedule exposes the following paremeters:

* WG tile size: [256, 256]
* SG tile size: [32, 32]
* K tile size: 32
* Load tile sizes for A and B: [16, 32], [16, 32]
* Prefetch tile sizes for A and B: [8, 32], [8, 16]
* Number of prefetch steps: 1

The following necessary parameter values can be inferred from the above:

* A, B, C tile `sg_layout`
* A, B prefetch `sg_layout`
* Correct number of threads in `gpu.launch`

These values can be computed dynamically in Python, for example.

## Future work

* Currently there's no mechanism to propagate xegpu layouts at WG level, e.g., from the anchor DPAS op to elementwise post-ops. Thus one has to manually attach the output ("C") layout to every elementwise op using the `transform.xegpu.set_op_layout_attr` transform op. Handling various pre/post op configurations in the transform schedule in this manner gets tedious. There exists `xegpu-propagate-layout` pass but it is applied later in the pipeline at instruction level, i.e. after `xegpu-wg-to-sg-distribute` and `xegpu-blocking` passes.
