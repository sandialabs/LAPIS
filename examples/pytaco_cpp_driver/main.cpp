#define LAPIS_CPP_DRIVER

#include <iostream>
#include <fstream>
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/SparseTensorRuntime.h"
#include "spmv.hpp"

// Helper to turn a std::vector into a rank-1 StridedMemRefType
// (it's a deep copy, so the result owns the memory)
template<typename T>
StridedMemRefType<T, 1> vectorToStridedMemRef(const std::vector<T>& vec)
{
  StridedMemRefType<T, 1> smr;
  smr.data = new T[vec.size()];
  smr.basePtr = smr.data;
  for(size_t i = 0; i < vec.size(); i++)
    smr.data[i] = vec[i];
  smr.offset = 0;
  smr.sizes[0] = vec.size();
  smr.strides[0] = 1;
  return smr;
}

/* Notes on newSparseTensor inputs (common for any dimension, any format)
 *
 * lvlSizes: number of tensor dimensions (equivalently, number of storage levels)
 * lvlTypesRef: storage type of each dim (sparse vs. dense)
 * lvl2dim: mapping between storage levels (of the data structure) and dimension (of the abstract tensor)
       For example:
       - in both CRS and CSC, level 0 is the offsets, and level 1 is the entries.
       - in any dense 2D tensor, level 0, 1 are the slow, fast varying dimensions respectively.
 * dim2lvl: the inverse of lvl2dimRef mapping.
 */

// Read a sparse tensor in as CRS format,
// from a FROSTT formatted file.
void* readCRS(const char* filename)
{
  auto Dense = mlir::sparse_tensor::DimLevelType::Dense;
  auto Compressed = mlir::sparse_tensor::DimLevelType::Compressed;
  auto kIndex = mlir::sparse_tensor::OverheadType::kIndex;
  auto kF32 = mlir::sparse_tensor::PrimaryType::kF32;
  std::vector<uint64_t> dims;
  readSparseTensorShape(const_cast<char*>(filename), &dims);
  if(dims.size() != 2) {
    throw std::invalid_argument("readCRS: the provided FROSTT file does not contain a rank-2 tensor.");
  }
  void* Areader = createSparseTensorReader(const_cast<char*>(filename));
  auto lvlSizesRef = vectorToStridedMemRef(std::vector<index_type>({dims[0], dims[1]}));
  auto lvlTypesRef = vectorToStridedMemRef(std::vector<mlir::sparse_tensor::DimLevelType>({Dense, Compressed}));
  auto lvl2dimRef = vectorToStridedMemRef(std::vector<index_type>({0, 1}));
  auto dim2lvlRef = vectorToStridedMemRef(std::vector<index_type>({0, 1}));
  void* A = _mlir_ciface_newSparseTensorFromReader(
      Areader, &lvlSizesRef, &lvlTypesRef,
      &lvl2dimRef, &dim2lvlRef, kIndex, kIndex, kF32);
  // Clean up reader
  delSparseTensorReader(Areader);
  std::cout << "Read in CRS matrix from \"" << filename << "\" with dimensions " << dims[0] << "x" << dims[1] << "\n";
  return A;
}

// Read a dense vector from a FROSTT formatted file.
void* readVector(const char* filename)
{
  auto Dense = mlir::sparse_tensor::DimLevelType::Dense;
  auto Compressed = mlir::sparse_tensor::DimLevelType::Compressed;
  auto kIndex = mlir::sparse_tensor::OverheadType::kIndex;
  auto kF32 = mlir::sparse_tensor::PrimaryType::kF32;
  std::vector<uint64_t> dims;
  readSparseTensorShape(const_cast<char*>(filename), &dims);
  if(dims.size() != 1) {
    throw std::invalid_argument("readVector: the provided FROSTT file does not contain a rank-1 tensor.");
  }
  void* reader = createSparseTensorReader(const_cast<char*>(filename));
  auto lvlSizesRef = vectorToStridedMemRef(std::vector<index_type>({dims[0]}));
  auto lvlTypesRef = vectorToStridedMemRef(std::vector<mlir::sparse_tensor::DimLevelType>({Dense}));
  auto lvl2dimRef = vectorToStridedMemRef(std::vector<index_type>({0}));
  auto dim2lvlRef = vectorToStridedMemRef(std::vector<index_type>({0}));
  void* vec = _mlir_ciface_newSparseTensorFromReader(
      reader, &lvlSizesRef, &lvlTypesRef,
      &lvl2dimRef, &dim2lvlRef, kIndex, kIndex, kF32);
  // Clean up reader
  delSparseTensorReader(reader);
  std::cout << "Read in vector from \"" << filename << "\" with " << dims[0] << " elements.\n";
  return vec;
}

// Write any kind of tensor to a FROSTT file.
void writeTensor(void* tensor, const char* filename)
{
  std::cout << "Writing a sparse tensor to \"" << filename << "\"\n";
  // Need to convert "sparse tensor storage" (STS) format to COO format for writing.
  uint64_t rank;
  uint64_t nnz;
  uint64_t* shape;
  float* values;
  uint64_t* indices;
  convertFromMLIRSparseTensorF32(tensor, &rank, &nnz, &shape, &values, &indices);
  std::ofstream os(filename);
  os << "# Extended FROSTT format:\n";
  os << "# rank number-non-zero-elements\n";
  os << "# dimension-sizes\n";
  os << rank << " " << nnz << "\n";
  for(uint64_t i = 0; i < rank; i++)
    os << shape[i] << " ";
  os << '\n';
  for(uint64_t i = 0; i < nnz; i++)
  {
    for(uint64_t j = 0; j < rank; j++)
    {
      os << indices[i * rank + j] << " ";
    }
    os << values[i] << "\n";
  }
  std::cout << "Wrote tensor with " << nnz << " elements and dimensions ";
  for(uint64_t i = 0; i < rank; i++)
  {
    std::cout << shape[i];
    if(i != rank - 1)
      std::cout << "x";
  }
  std::cout << " to \"" << filename << "\"\n";
}

int main()
{
  lapis_initialize();
  {
    // Compute c = A*b where:
    //   A is a CRS matrix
    //   b, c are dense vectors
    void* A = readCRS("A.tns");
    void* b = readVector("b.tns");
    void* c = pytaco_main((int8_t*) A, (int8_t*) b);
    writeTensor(c, "c.tns");
  }
  lapis_finalize();
  return 0;
}

