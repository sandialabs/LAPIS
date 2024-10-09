/*===-- torch-mlir-c/Registration.h - Registration functions  -----*- C -*-===*/

#ifndef LAPIS_C_REGISTRATION_H
#define LAPIS_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void lapisRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // LAPIS_C_REGISTRATION_H
