#ifndef LAPIS_INITALLKOKKOSPASSES_H
#define LAPIS_INITALLKOKKOSPASSES_H

#include "mlir/InitAllPasses.h"
#include "lapis/Dialect/Kokkos/Pipelines/Passes.h"

namespace mlir {

inline void registerAllKokkosPasses() {
  // Dialect pipelines
  kokkos::registerKokkosPipelines();
}

} // namespace mlir

#endif // LAPIS_INITALLKOKKOSPASSES_H_
