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
declare_mlir_python_sources(LapisPythonExtensions)

declare_mlir_python_sources(LapisPythonSources.kokkos_backend
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lapis"
  ADD_TO_PARENT LapisPythonSources
  SOURCES
    KokkosBackend.py
)

#declare_mlir_python_sources(LapisPythonSources.tools
#  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lapis"
#  ADD_TO_PARENT LapisPythonSources
#  SOURCES
#    tools/mlir_pytaco_api.py
#    tools/mlir_pytaco_utils.py
#    tools/testing_utils.py
#    tools/mlir_pytaco.py
#    tools/mlir_pytaco_io.py
#    tools/mlir_sparse_compiler.py
#)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(LapisPythonExtensions.Main
  MODULE_NAME _lapis
  ADD_TO_PARENT LapisPythonExtensions
  SOURCES
    LapisModule.cpp
  EMBED_CAPI_LINK_LIBS
    LapisCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

target_include_directories(LapisPythonExtensions.Main INTERFACE
  $<BUILD_INTERFACE:${LAPIS_SOURCE_DIR}/mlir/include>
  $<BUILD_INTERFACE:${LAPIS_BINARY_DIR}/mlir/include>)
target_link_libraries(LapisPythonExtensions.Main INTERFACE MLIRPythonExtension.Core LapisCAPI)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  # TODO: Reduce dependencies. We need ExecutionEngine and a bunch of passes
  # for the reference backend, but logically they can be separate. But seemingly
  # the only way to handle that is to create a separate mlir python package
  # tree, which seems excessive.
  LapisPythonSources
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  LapisPythonExtensions
)

add_mlir_python_common_capi_library(LapisAggregateCAPI
  INSTALL_COMPONENT LapisPythonModules
  INSTALL_DESTINATION python_packages/lapis/_mlir_libs
  OUTPUT_DIRECTORY "${LAPIS_PYTHON_PACKAGES_DIR}/lapis/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

target_link_libraries(
  LapisAggregateCAPI
  PUBLIC
  LapisCAPI
)

add_mlir_python_modules(LapisPythonModules
  ROOT_PREFIX "${LAPIS_PYTHON_PACKAGES_DIR}/lapis/lapis"
  INSTALL_PREFIX "python_packages/lapis"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    LapisAggregateCAPI
  )
