include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(LAPIS_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lapis")


# We vendor our own MLIR instance in the `lapis` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=lapis.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(LapisPythonSources)

declare_mlir_python_sources(LapisPythonSources.kokkos_backend
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lapis"
  ADD_TO_PARENT LapisPythonSources
  SOURCES
    KokkosBackend.py
)

set(_source_components
  LapisPythonSources
)

add_mlir_python_modules(LapisPythonModules
  ROOT_PREFIX "${LAPIS_PYTHON_PACKAGES_DIR}/lapis/lapis"
  INSTALL_PREFIX "python_packages/lapis"
  DECLARED_SOURCES ${_source_components}
  )
