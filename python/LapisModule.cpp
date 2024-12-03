#include "lapis-c/Dialects.h"
#include "lapis-c/EmitKokkos.h"
#include "lapis-c/Registration.h"
#include "lapis/InitAllKokkosPasses.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

void lapisRegisterAllPasses() {
  printf("*************** registerAllKokkosPasses() "
         "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  mlir::registerAllKokkosPasses();
}

namespace py = pybind11;

using namespace mlir::python;

PYBIND11_MODULE(_lapis, m) {
  lapisRegisterAllPasses();

  m.doc() =
      "LAPIS extension for Python (MLIR pipeline + translation to Kokkos C++)";

  m.def(
      "emit_kokkos",
      [](const void *modulePtr, const char *cxxSourceFile,
         const char *pySourceFile, bool isLastKernel) {
        MlirModule module = {modulePtr};
        MlirLogicalResult status =
            lapisEmitKokkos(module, cxxSourceFile, pySourceFile, isLastKernel);
        if (mlirLogicalResultIsFailure(status))
          throw std::runtime_error(
              "Failure while lowering MLIR to Kokkos C++ source code.");
      },
      py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"),
      py::arg("is_final_kernel"),
      "Emit Kokkos C++ and Python wrappers for the given module, and throw a "
      "RuntimeError on failure.");

  m.def(
      "lower_and_emit_kokkos",
      [](const char *moduleText, const char *cxxSourceFile,
         const char *pySourceFile, bool isLastKernel) {
        MlirLogicalResult status = lapisLowerAndEmitKokkos(
            moduleText, cxxSourceFile, pySourceFile, isLastKernel);
        if (mlirLogicalResultIsFailure(status))
          throw std::runtime_error(
              "Failure while lowering MLIR to Kokkos C++ source code.");
      },
      py::arg("module_text"), py::arg("cxx_source_file"),
      py::arg("py_source_file"), py::arg("is_final_kernel"),
      "Emit Kokkos C++ and Python wrappers for the given module, and throw a "
      "RuntimeError on failure.");
}
