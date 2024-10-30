//===- PartTensorRuntime.cpp - PartTensor runtime support lib ---------===//
//
// **** This file has been modified from its original in llvm-project ****
// Original file was mlir/lib/ExecutionEngine/SparseTensorRuntime.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime support library for
// manipulating paritioned sparse tensors from MLIR.  More specifically, it
// provides C-API wrappers so that MLIR-generated code can call into the C++
// runtime support library.  The functionality provided in this library is meant
// to simplify benchmarking, testing, and debugging of MLIR code operating
// on sparse tensors.  However, the provided functionality is **not**
// part of core MLIR itself.
//
// The following memory-resident partitioned sparse storage schemes are
// supported:
//
// (a) A coordinate scheme for temporarily storing and lexicographically
//     sorting a sparse tensor by index (SparseTensorCOO).
//
//  // TODO: support other things supported by SparseTensor.
//
// The following external formats are supported:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// Two public APIs are supported:
//
// (I) Methods operating on MLIR buffers (memrefs) to interact with partitioned
//     sparse tensors. These methods should be used exclusively by MLIR
//     compiler-generated code.
//
// (II) Methods that accept C-style data structures to interact with partitioned
//      sparse tensors. These methods can be used by any external runtime that
//      wants to interact with MLIR compiler-generated code.
//
// In both cases (I) and (II), the SparseTensorStorage format is externally
// only visible as an opaque pointer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/SparseTensorRuntime.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <cstdint>

#include "lapis/ExecutionEngine/PartTensor/Storage.h"
#include "lapis/ExecutionEngine/PartTensorRuntime.h"

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/File.h"
#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#include <cstring>
#include <numeric>

using namespace mlir::part_tensor;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
//
// Implementation details for public functions, which don't have a good
// place to live in the C++ library this file is wrapping.
//
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
//
// Utilities for manipulating `StridedMemRefType`.
//
//===----------------------------------------------------------------------===//

// We shouldn't need to use `detail::safelyEQ` here since the `1` is a literal.
#define ASSERT_NO_STRIDE(MEMREF)                                               \
  do {                                                                         \
    assert((MEMREF) && "Memref is nullptr");                                   \
    assert(((MEMREF)->strides[0] == 1) && "Memref has non-trivial stride");    \
  } while (false)

// All our functions use `uint64_t` for ranks, but `StridedMemRefType::sizes`
// uses `int64_t` on some platforms.  So we explicitly cast this lookup to
// ensure we get a consistent type, and we use `checkOverflowCast` rather
// than `static_cast` just to be extremely sure that the casting can't
// go awry.  (The cast should aways be safe since (1) sizes should never
// be negative, and (2) the maximum `int64_t` is smaller than the maximum
// `uint64_t`.  But it's better to be safe than sorry.)
#define MEMREF_GET_USIZE(MEMREF)                                               \
  detail::checkOverflowCast<uint64_t>((MEMREF)->sizes[0])

#define ASSERT_USIZE_EQ(MEMREF, SZ)                                            \
  assert(detail::safelyEQ(MEMREF_GET_USIZE(MEMREF), (SZ)) &&                   \
         "Memref size mismatch")

#define MEMREF_GET_PAYLOAD(MEMREF) ((MEMREF)->data + (MEMREF)->offset)

} // anonymous namespace

extern "C" {

// Assume index_type is in fact uint64_t, so that _mlir_ciface_newSparseTensor
// can safely rewrite kIndex to kU64.  We make this assertion to guarantee
// that this file cannot get out of sync with its header.
static_assert(std::is_same<index_type, uint64_t>::value,
              "Expected index_type == uint64_t");

void _mlir_ciface_getPartitions( // NOLINT
    StridedMemRefType<index_type, 1> *partsMemRef, void *tensor) {
  std::vector<index_type> *parts;
  static_cast<PartTensorStorageBase *>(tensor)->getPartitions(&parts);
  aliasIntoMemref(parts->size(), parts->data(), *partsMemRef);
}

index_type _mlir_ciface_getNumPartitions(void *tensor) {
  return static_cast<PartTensorStorageBase *>(tensor)->getNumPartitions();
}

void *_mlir_ciface_getSlice(void *tensor,
                            StridedMemRefType<index_type, 1> *partSpec) {
  return static_cast<PartTensorStorageBase *>(tensor)->getSlice(
      llvm::ArrayRef<index_type>(partSpec->data + partSpec->offset,
                                 partSpec->sizes[0]));
}
void _mlir_ciface_setSlice(void *tensor,
                           StridedMemRefType<index_type, 1> *partSpec,
                           void *spTensor) {
  static_cast<PartTensorStorageBase *>(tensor)->setSlice(
      llvm::ArrayRef<index_type>(partSpec->data + partSpec->offset,
                                 partSpec->sizes[0]),
      static_cast<SparseTensorStorageBase *>(spTensor));
}
extern void *snl_utah_spadd_dense_f32(void *tensor, void *spTensor);

void _mlir_ciface_updateSlice(void *partTensor,
                              StridedMemRefType<index_type, 1> *partSpec,
                              void *spTensor) {
  // For now it only works on dense.
  auto *oldVal = static_cast<SparseTensorStorageBase *>(
      _mlir_ciface_getSlice(partTensor, partSpec));
  auto *newVal = static_cast<SparseTensorStorageBase *>(spTensor);
  std::vector<float> *valuesVector;
  std::vector<float> *newValuesVector;
  oldVal->getValues(&valuesVector);
  newVal->getValues(&newValuesVector);
  for (auto elem : llvm::zip(*valuesVector, *newValuesVector)) {
    std::get<0>(elem) += std::get<1>(elem);
  }
}
} // extern "C"

#undef MEMREF_GET_PAYLOAD
#undef ASSERT_USIZE_EQ
#undef MEMREF_GET_USIZE
#undef ASSERT_NO_STRIDE

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
