set(LLVM_TARGET_DEFINITIONS KokkosDialect.td)
mlir_tablegen(Kokkos.h.inc -gen-op-decls)
mlir_tablegen(Kokkos.cpp.inc -gen-op-defs)
mlir_tablegen(KokkosEnums.h.inc -gen-enum-decls)
mlir_tablegen(KokkosEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KokkosDialect.h.inc -gen-dialect-decls -dialect=kokkos)
mlir_tablegen(KokkosDialect.cpp.inc -gen-dialect-defs -dialect=kokkos)
# We don't use any custom types, but uncomment this if we do later on.
# (and include KokkosTypes.h.inc in KokkosDialect.h)
#mlir_tablegen(KokkosTypes.h.inc -gen-typedef-decls -typedefs-dialect=kokkos)
#mlir_tablegen(KokkosTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=kokkos)
add_public_tablegen_target(MLIRKokkosIncGen)
add_dependencies(mlir-headers MLIRKokkosIncGen)
add_mlir_doc(Kokkos Kokkos Dialects/ -gen-dialect-doc)

