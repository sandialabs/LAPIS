#ifndef LAPIS_INITALLTRANSLATIONS_H
#define LAPIS_INITALLTRANSLATIONS_H

#include "mlir/InitAllTranslations.h"

namespace mlir {

void registerToKokkosTranslation();

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllKokkosTranslations() {
  registerAllTranslations();
  static bool initOnce = []() {
    registerToKokkosTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace mlir

#endif // LAPIS_INITALLTRANSLATIONS_H
