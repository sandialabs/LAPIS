//===- Passes.h - Part tensor pipeline entry points -----------*- C++ -*-===//
//
// **** This file has been modified from its original in llvm-project ****
// Original file was mlir/include/mlir/Dialect/SparseTensor/Pipelines/Passes.h
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PARTTENSOR_PIPELINES_PASSES_H_
#define MLIR_DIALECT_PARTTENSOR_PIPELINES_PASSES_H_

#include "lapis/Dialect/Kokkos/Pipelines/Passes.h"
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Pass/PassOptions.h"

// for SparseParallelizationStrategy
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace part_tensor {
using mlir::kokkos::LapisCompilerOptions;
} // namespace part_tensor
} // namespace mlir

#endif // MLIR_DIALECT_PARTTENSOR_PIPELINES_PASSES_H_
