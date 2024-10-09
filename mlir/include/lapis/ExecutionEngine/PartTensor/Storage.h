//===- Storage.h -    Partitioned tensor representation       ---*- C++ -*-===//
//
// This file is part of the lightweight runtime support library for partitioned
// sparse tensor manipulations.  The functionality of the support library is
// meant to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
// Ideally we would split the storage classes and enumerator classes
// into separate files, to improve legibility.  But alas: because these
// are template-classes, they must therefore provide *definitions* in the
// header; and those definitions cause circular dependencies that make it
// impossible to split the file up along the desired lines.  (We could
// split the base classes from the derived classes, but that doesn't
// particularly help improve legibility.)
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_PARTTENSOR_STORAGE_H
#define MLIR_EXECUTIONENGINE_PARTTENSOR_STORAGE_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"
#include "mlir/ExecutionEngine/SparseTensor/Storage.h"
#include "mlir/ExecutionEngine/SparseTensorRuntime.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace mlir {
// TODO: part_tensor need to have it's own namespace.
namespace part_tensor {

using llvm::ArrayRef;
using std::unique_ptr;

/**
 * Point in num-dims-dimensional-cartesian space.
 */
template <uint64_t num_dims, class index_type = size_t>
using PartPoint = std::array<index_type, num_dims>;

/**
 * Pair of point in num-dims-dimensional-cartesian space, to denote a
 * hyper-rectangle.
 */
template <uint64_t num_dims, class indexTy = size_t,
          class PartPointTy = PartPoint<num_dims, indexTy>>
using PartSpec = std::pair<PartPointTy, PartPointTy>;

/**
 * Set of hyper-rectangles.
 */
template <uint64_t num_dims, class indexTy = size_t,
          class PartPointTy = PartPoint<num_dims, indexTy>>
using PartitionPlan = std::vector<PartSpec<num_dims, indexTy, PartPointTy>>;

/// ABC so that c api can call into template specializations.
class PartTensorStorageBase {
public:
  virtual ~PartTensorStorageBase() = default;
  virtual void getPartitions(std::vector<index_type> **) = 0;
  virtual index_type getNumPartitions() = 0;
  virtual void *getSlice(llvm::ArrayRef<index_type> partSpec) = 0;
  virtual void setSlice(llvm::ArrayRef<index_type> partSpec,
                        SparseTensorStorageBase *spTensor) = 0;
};

/// A memory-resident sparse tensor using a storage scheme based on
/// per-dimension sparse/dense annotations.  This data structure provides
/// a bufferized form of a sparse tensor type.  In contrast to generating
/// setup methods for each differently annotated sparse tensor, this
/// method provides a convenient "one-size-fits-all" solution that simply
/// takes an input tensor and annotations to implement all required setup
/// in a general manner.
template <typename P = uint64_t, typename I = uint64_t, typename V = float>
class PartTensorStorage : public PartTensorStorageBase {

public:
  /// The called is resposible to keep spCOO alive for the lifetime of this
  /// partTensor.
  PartTensorStorage(uint64_t nParts, const uint64_t *partDataPtr,
                    std::vector<unique_ptr<SparseTensorCOO<V>>> &&spCOOs,
                    std::vector<SparseTensorStorage<P, I, V> *> partTensors_)
      : partData(partDataPtr, partDataPtr + nParts), parts(std::move(spCOOs)),
        partTensors(partTensors_) {}

  static PartTensorStorage<P, I, V> *
  newFromCOO(uint64_t nParts, const uint64_t *partData, uint64_t dimRank,
             const uint64_t *dimShape, const DimLevelType *lvlTypes,
             const SparseTensorCOO<V> *spCOO);

  ~PartTensorStorage() = default;
  void getPartitions(std::vector<index_type> **partData) override {
    *partData = &this->partData;
  }
  index_type getNumPartitions() override {
    return 2 * parts.size() * getRank();
  }
  auto getRank() { return parts[0]->getRank(); }
  void *getSlice(llvm::ArrayRef<index_type> partSpec) override {
    auto partSpecSize = 2 * getRank();
    assert(std::size(partSpec) == partSpecSize);
    for (auto i : llvm::seq(0ul, std::size(parts))) {
      auto loOffset = 2 * i * getRank();
      ArrayRef curPartSpec(partData.data() + loOffset, partSpecSize);
      if (curPartSpec == partSpec) {
        return partTensors[i];
      }
    }
    assert(0 &&
           "We currently don't support union or intersection of partitions");
    return nullptr;
  }
  void setSlice(llvm::ArrayRef<index_type> partSpec,
                SparseTensorStorageBase *spTensor) override {
    auto partSpecSize = 2 * getRank();
    assert(std::size(partSpec) == partSpecSize);
    for (auto i : llvm::seq(0ul, std::size(parts))) {
      auto loOffset = 2 * i * getRank();
      ArrayRef curPartSpec(partData.data() + loOffset, partSpecSize);
      if (curPartSpec == partSpec) {
        partTensors[i] = static_cast<SparseTensorStorage<P, I, V> *>(spTensor);
        return;
      }
    }
    assert(0 &&
           "We currently don't support union or intersection of partitions");
  }
  auto &getParts() { return partTensors; }

protected:
  std::vector<index_type> partData;
  const std::vector<unique_ptr<SparseTensorCOO<V>>> parts;
  std::vector<SparseTensorStorage<P, I, V> *> partTensors;
};

template <typename T>
bool inRegion(T loPoint, T hiPoint, T point) {
  // TODO: need to make sure T is not reference type. We expect only
  // std::span/llvm::ArrayRef to come in here.
  for (auto i : llvm::seq(0lu, std::size(point))) {
    if (point[i] < loPoint[i] || point[i] >= hiPoint[i]) {
      return false;
    }
  }
  return true;
}

template <typename P, typename I, typename V>
PartTensorStorage<P, I, V> *PartTensorStorage<P, I, V>::newFromCOO(
    uint64_t partDataLength, const uint64_t *partData, uint64_t dimRank,
    const uint64_t *dimShape, const DimLevelType *lvlTypes,
    const SparseTensorCOO<V> *spCOO) {
  using namespace mlir::part_tensor;
  using llvm::ArrayRef;
  assert(partData && "Got nullptr for partition shape");
  assert(dimShape && "Got nullptr for dimension shape");
  assert(partDataLength > 0 && "Got zero for partition data");
  assert(dimRank > 0 && "Got zero for dimension rank");
  std::vector<uint64_t> dimSizes(dimRank);
  unsigned long numPartitions = partDataLength / (dimRank * 2);
  assert(partDataLength % (dimRank * 2) == 0 &&
         "Partition data len must be a multiple of dimension rank");

  std::vector<unique_ptr<SparseTensorCOO<V>>> parts;
  std::vector<SparseTensorStorage<P, I, V> *> partTensors;
  parts.reserve(numPartitions);
  partTensors.resize(numPartitions);
  // For now map 1-1 lvl to dim
  uint64_t lvl2dim[2] = {0, 1};
  for (auto i : llvm::seq(0lu, numPartitions)) {
    auto loOffset = 2 * i * dimRank;
    auto hiOffset = (2 * i + 1) * dimRank;
    ArrayRef loPoint(partData + loOffset, dimRank);
    ArrayRef hiPoint(partData + hiOffset, dimRank);
    // calculate x - y for every x in hipoint and every y in lopoint.
    std::vector<I> partShape(dimRank);
    std::transform(std::begin(hiPoint), std::end(hiPoint), std::begin(loPoint),
                   std::begin(partShape), std::minus<I>());
    assert(llvm::all_of(partShape, [](auto i) { return i > 0; }) &&
           "Patition shape must be positive");
    // for (auto i : partShape) {
    //   std::cout << i << ", ";
    // }
    // std::cout << "\n";
    parts.emplace_back(std::make_unique<SparseTensorCOO<V>>(partShape));

    llvm::for_each(spCOO->getElements(), [&](auto e) {
      std::vector<I> newCoords(dimRank);
      std::transform((e.coords), (e.coords + dimRank), std::begin(loPoint),
                     std::begin(newCoords), std::minus<I>());

      if (inRegion(loPoint, hiPoint, ArrayRef(e.coords, dimRank))) {
        parts[i]->add(newCoords, e.value);
      }
    });
    // TODO: get rid of parts when all code is migrated to partTensors

    partTensors[i] = SparseTensorStorage<P, I, V>::newFromCOO(
        dimRank, partShape.data(), dimRank /*lvlRank*/, lvlTypes, lvl2dim,
        *parts[i]);
  }
  return new PartTensorStorage<P, I, V>(
      partDataLength, partData, std::move(parts), std::move(partTensors));
}

} // namespace part_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_PARTTENSOR_STORAGE_H
