#include "lapis/InitAllKokkosPasses.h"
#include "lapis-c/Dialects.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/CAPI/Registration.h"

void lapisRegisterAllPasses() {
  mlir::registerAllKokkosPasses();
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Kokkos, kokkos, mlir::kokkos::KokkosDialect)
