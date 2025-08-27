set(LLVM_TARGET_DEFINITIONS FFIExtensionOps.td)
mlir_tablegen(FFIExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(FFIExtensionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTransformDialectFFIExtensionOpsIncGen)

add_mlir_doc(FFIExtensionOps FFIExtensionOps Dialects/ -gen-op-doc)
