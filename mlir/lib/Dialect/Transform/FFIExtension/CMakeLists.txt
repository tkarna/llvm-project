add_mlir_dialect_library(MLIRTransformFFIExtension
  FFIExtension.cpp
  FFIExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectFFIExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
)
