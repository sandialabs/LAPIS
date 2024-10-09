#ifndef LAPIS_C_DIALECTS_H
#define LAPIS_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Kokkos, kokkos);

#ifdef __cplusplus
}
#endif

#endif  // LAPIS_C_DIALECTS_H
