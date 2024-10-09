//===- Passes.h - Part tensor pass entry points ---------------*- C++ -*-===//
//
// This header file defines prototypes of all sparse tensor passes.
//
// In general, this file takes the approach of keeping "mechanism" (the
// actual steps of applying a transformation) completely separate from
// "policy" (heuristics for when and where to apply transformations).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

enum class PartTensorDistBackend {
  kNone, // serialize part tensor execution
  kMPI,  // use MPI for part tensor execution
  kKRS   // use kokkos remote spaces
};

#define GEN_PASS_DECL
#include "mlir/Dialect/PartTensor/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// The PartTensorConversion pass.
//===----------------------------------------------------------------------===//

/// Part tensor type converter into an opaque pointer.
class PartTensorTypeToPtrConverter : public TypeConverter {
public:
  PartTensorTypeToPtrConverter();
};

void populatePartTensorConversionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PartTensorDistBackend backend);

std::unique_ptr<Pass> createPartTensorConversionPass();
std::unique_ptr<Pass> createPartTensorConversionPass(PartTensorDistBackend);

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/PartTensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES_H_
