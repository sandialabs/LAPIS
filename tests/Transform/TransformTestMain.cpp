#include "TestSupport.h"

namespace mlir::lapis {

void runContractionPlannerTests(test::TestContext &context);
void runEinsumTests(test::TestContext &context);

} // namespace mlir::lapis

int main() {
  mlir::lapis::test::TestContext context;
  mlir::lapis::runContractionPlannerTests(context);
  mlir::lapis::runEinsumTests(context);
  return context.getExitCode();
}
