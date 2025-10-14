module attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>, %niter: index) -> f64 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %A_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
    %B_gpu = gpu.alloc () : memref<4096x4096xf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
    %C_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>

    // Warm-up
    scf.for %i = %c0 to %c4 step %c1 {
      func.call @payload(%A_gpu, %B_gpu, %C_gpu) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> ()
    }

    // Measure execution time
    %tic = call @rtclock() : () -> f64
    scf.for %i = %c0 to %niter step %c1 {
      func.call @payload(%A_gpu, %B_gpu, %C_gpu) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> ()
    }
    %toc = call @rtclock() : () -> f64
    %duration = arith.subf %toc, %tic : f64
    %niter_i64 = arith.index_cast %niter : index to i64
    %niter_f64 = arith.sitofp %niter_i64 : i64 to f64
    %time = arith.divf %duration, %niter_f64 : f64

    // Calculate final solution.
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
    func.call @payload(%A_gpu, %B_gpu, %C_gpu) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> ()

    gpu.memcpy %C, %C_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc %A_gpu : memref<4096x4096xf16>
    gpu.dealloc %B_gpu : memref<4096x4096xf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf32>
    return %time : f64
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %niter = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %c1_f32 = arith.constant 1.0 : f32
    %c0_f32 = arith.constant 0.0 : f32
    %c1_f16 = arith.constant 1.0 : f16
    %c2_f16 = arith.constant 2.0 : f16
    %A = memref.alloc() : memref<4096x4096xf16>
    %B = memref.alloc() : memref<4096x4096xf16>
    %C = memref.alloc() : memref<4096x4096xf32>
    // Intialize A,B,C
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c1_f16, %A[%i, %j] : memref<4096x4096xf16>
        memref.store %c2_f16, %B[%i, %j] : memref<4096x4096xf16>
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Call kernel
    %time = call @test(%A, %B, %C, %niter) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>, index) -> (f64)

    // Print a row of C to check correctness
    %C_row_0_gpu  = memref.subview %C[0, 0][1, 4096][1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset:0>>
    %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x4096xf32, strided<[4096, 1], offset: 0>> to memref<*xf32>
    call @printMemrefF32(%C_row_0_cast_gpu) : (memref<*xf32>) -> ()

    // Print timing
    vector.print str "Average time in kernel: "
    vector.print %time : f64

    memref.dealloc %A : memref<4096x4096xf16>
    memref.dealloc %B : memref<4096x4096xf16>
    memref.dealloc %C : memref<4096x4096xf32>
    return
  }
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %3 = linalg.matmul ins(%0, %1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
