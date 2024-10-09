set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Kokkos)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Kokkos)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Kokkos)
add_public_tablegen_target(MLIRKokkosPassIncGen)

add_mlir_doc(Passes KokkosPasses ./ -gen-pass-doc)
