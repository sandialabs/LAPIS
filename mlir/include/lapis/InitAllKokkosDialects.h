#ifndef LAPIS_INITALLDIALECTS_H_
#define LAPIS_INITALLDIALECTS_H_

#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"

namespace mlir {

/// Add all the MLIR dialects to the provided registry.
inline void registerAllKokkosDialects(DialectRegistry &registry) {
  registerAllDialects(registry);
  // clang-format off
  registry.insert<part_tensor::PartTensorDialect>();
  // clang-format on
}

/// Append all the MLIR dialects to the registry contained in the given context.
inline void registerAllKokkosDialects(MLIRContext &context) {
  DialectRegistry registry;
  registerAllKokkosDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mlir

#endif // LAPIS_INITALLDIALECTS_H_
