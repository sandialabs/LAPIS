//===- lapis-opt.cpp - LAPIS pass Driver -------------------------===//
//
// **** This file has been modified from its original in llvm-project ****
// Original file was mlir/tools/mlir-opt/mlir-opt.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Pipelines/Passes.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "lapis/Dialect/PartTensor/IR/PartTensor.h"
#include "lapis/Dialect/PartTensor/Pipelines/Passes.h"
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"
#endif
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  // lapis-opt is intended to drive only passes from the custom
  // LAPIS dialects (Kokkos and PartTensor), not builtin dialects.
  // Use mlir-opt for those passes.
  DialectRegistry registry;
  registry.insert<
#ifdef ENABLE_PART_TENSOR
      mlir::part_tensor::PartTensorDialect, 
#endif
      mlir::LLVM::LLVMDialect, mlir::vector::VectorDialect,
      mlir::bufferization::BufferizationDialect, mlir::linalg::LinalgDialect,
      mlir::sparse_tensor::SparseTensorDialect, mlir::tensor::TensorDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::ml_program::MLProgramDialect,
      mlir::kokkos::KokkosDialect>();

  // Have to also register dialect extensions.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
  linalg::registerAllDialectInterfaceImplementations(registry);
  linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  ml_program::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register LAPIS pipelines and passes
#ifdef ENABLE_PART_TENSOR
  mlir::registerPartTensorPasses();
#endif

  kokkos::registerKokkosPipelines();
  mlir::registerKokkosPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LAPIS/MLIR pass driver\n", registry));
}
