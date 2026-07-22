set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LapisTransform)
add_public_tablegen_target(MLIRLapisTransformPassIncGen)

add_mlir_doc(Passes LapisTransformPasses ./ -gen-pass-doc)
