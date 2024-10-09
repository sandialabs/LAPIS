#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "lapis-c/Dialects.h"
#include "lapis-c/Registration.h"
#include "lapis-c/EmitKokkos.h"
#include "lapis/InitAllKokkosPasses.h"

void lapisRegisterAllPasses() {
  printf("*************** registerAllKokkosPasses() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  mlir::registerAllKokkosPasses();
}

namespace py = pybind11;

using namespace mlir::python;

PYBIND11_MODULE(_lapis, m) {
  lapisRegisterAllPasses();

  m.doc() = "LAPIS extension for Python (MLIR pipeline + translation to Kokkos C++)";

  m.def("emit_kokkos",
          [](const void* modulePtr, const char* cxxSourceFile, const char* pySourceFile) {
            MlirModule module = {modulePtr};
            MlirLogicalResult status =
                lapisEmitKokkos(module, cxxSourceFile, pySourceFile);
            if (mlirLogicalResultIsFailure(status))
              throw std::runtime_error("Failure while raising MLIR to Kokkos C++ source code.");
          },
          py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"),
          "Emit Kokkos C++ and Python wrappers for the given module, and throw a RuntimeError on failure.");

  m.def("emit_kokkos_sparse",
          [](const void* modulePtr, const char* cxxSourceFile, const char* pySourceFile, bool useHierarchical, bool isLastKernel) {
            MlirModule module = {modulePtr};
            MlirLogicalResult status =
                lapisEmitKokkosSparse(module, cxxSourceFile, pySourceFile, useHierarchical, isLastKernel);
            if (mlirLogicalResultIsFailure(status))
              throw std::runtime_error("Failure while raising MLIR to Kokkos C++ source code.");
          },
          py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"), py::arg("use_hierarchical"), py::arg("is_final_kernel"),
          "Emit Kokkos C++ and Python wrappers for the given sparse module, and throw a RuntimeError on failure.");
}

