cmake_minimum_required(VERSION 3.20.0)
project(lapis LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Turn on part tensor dialect by default
IF(NOT DEFINED LAPIS_ENABLE_PART_TENSOR)
  SET(LAPIS_ENABLE_PART_TENSOR ON)
ENDIF()

if(LAPIS_ENABLE_PART_TENSOR)
  add_compile_definitions(ENABLE_PART_TENSOR)
endif()

macro(lapis_add_llvm_external_project name identifier location)
  message(STATUS "Adding LLVM external project ${name} (${identifier}) -> ${location}")
  if(NOT EXISTS "${location}/CMakeLists.txt")
    message(FATAL_ERROR "External project location ${location} is not valid")
  endif()
  list(APPEND LLVM_EXTERNAL_PROJECTS ${name})
  list(REMOVE_DUPLICATES LLVM_EXTERNAL_PROJECTS)
  set(LLVM_EXTERNAL_${identifier}_SOURCE_DIR ${location} CACHE STRING "" FORCE)
  set(LLVM_EXTERNAL_PROJECTS ${LLVM_EXTERNAL_PROJECTS} CACHE STRING "" FORCE)
endmacro()

lapis_add_llvm_external_project(
  lapis-dialects
  LAPIS_DIALECTS
  ${CMAKE_CURRENT_SOURCE_DIR}/mlir)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(STATUS "LAPIS: Building against LLVM/MLIR installation")
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  # Where lapis-opt, lapis-translate are built (need this for Lit tests)
  set(LAPIS_TOOL_BIN ${CMAKE_BINARY_DIR}/bin)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  message(STATUS "LAPIS: Building in-tree as LLVM external project")
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(LAPIS_TOOL_BIN ${LLVM_BINARY_DIR}/bin)
endif()

# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()

  if(NOT LAPIS_PYTHON_PACKAGES_DIR)
    set(LAPIS_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  endif()
endif()

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)

set(LAPIS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LAPIS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(mlir)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

# This should be at the end after traversal of all subdirectories.
add_subdirectory(cmake)

enable_testing()
add_subdirectory(tests)

