#ifndef LAPIS_EMITKOKKOS_HPP
#define LAPIS_EMITKOKKOS_HPP

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Given the source code (ASCII text) for a linalg-level MLIR module,
// lower to Kokkos dialect and emit Kokkos source code.
MlirLogicalResult lapisLowerAndEmitKokkos(const char *moduleText,
                                          const char *cxxSourceFile,
                                          const char *pySourceFile,
                                          bool isLastKernel);

MlirLogicalResult lapisEmitKokkos(MlirModule module, const char *cxxSourceFile,
                                  const char *pySourceFile, bool isLastKernel);

#ifdef __cplusplus
}
#endif

#endif
