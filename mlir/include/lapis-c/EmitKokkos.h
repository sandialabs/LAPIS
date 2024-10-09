#ifndef LAPIS_EMITKOKKOS_HPP
#define LAPIS_EMITKOKKOS_HPP

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MlirLogicalResult lapisEmitKokkos(MlirModule module, const char* cxxSourceFile,
                                  const char* pySourceFile);

MlirLogicalResult lapisEmitKokkosSparse(MlirModule module,
                                        const char* cxxSourceFile,
                                        const char* pySourceFile,
                                        bool useHierarchical,
                                        bool isLastKernel);

#ifdef __cplusplus
}
#endif

#endif
