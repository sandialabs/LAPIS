#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstdint>
#include <unistd.h>
using exec_space = Kokkos::DefaultExecutionSpace;

extern "C" void lapis_initialize();
extern "C" void lapis_finalize();

// If building a CPP driver, we can use the original StridedMemRefType class from MLIR,
// so do not redefine it here.
#ifndef PYTACO_CPP_DRIVER
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
#endif

// If building a CPP driver, need to provide a version of
//_mlir_ciface_newSparseTensor() that takes enum types, not underlying integer types.
#ifdef PYTACO_CPP_DRIVER
int8_t* _mlir_ciface_newSparseTensor(
  StridedMemRefType<index_type, 1> *dimSizesRef,
  StridedMemRefType<index_type, 1> *lvlSizesRef,
  StridedMemRefType<int8_t, 1> *lvlTypesRef,
  StridedMemRefType<index_type, 1> *lvl2dimRef,
  StridedMemRefType<index_type, 1> *dim2lvlRef, int ptrTp,
  int indTp, int valTp, int action, int8_t* ptr) {
    return (int8_t*) _mlir_ciface_newSparseTensor(dimSizesRef, lvlSizesRef,
      reinterpret_cast<StridedMemRefType<DimLevelType, 1>*>(lvlTypesRef),
      lvl2dimRef, dim2lvlRef, (OverheadType) ptrTp, (OverheadType) indTp,
      (PrimaryType) valTp, (Action) action, ptr);
  }
#endif

template<typename V>
StridedMemRefType<typename V::value_type, V::rank> viewToStridedMemref(const V& v)
{
  static_assert(std::is_same_v<typename V::memory_space, Kokkos::HostSpace>, "Only Kokkos::Views in HostSpace can be converted to StridedMemRefType.");
  StridedMemRefType<typename V::value_type, V::rank> smr;
  smr.basePtr = v.data();
  smr.data = v.data();
  smr.offset = 0;
  for(int i = 0; i < V::rank; i++)
  {
    smr.sizes[i] = v.extent(i);
    smr.strides[i] = v.stride(i);
  }
  return smr;
}

template<typename V>
V stridedMemrefToView(const StridedMemRefType<typename V::value_type, V::rank>& smr)
{
  using Layout = typename V::array_layout;
  static_assert(std::is_same_v<typename V::memory_space, Kokkos::HostSpace>, "Can only convert a StridedMemRefType to a Kokkos::View in HostSpace.");
  if constexpr(std::is_same_v<Layout, Kokkos::LayoutStride>)
  {
    Layout layout(
    (0 < V::rank) ? smr.sizes[0] : 0U,
    (0 < V::rank) ? smr.strides[0] : 0U,
    (1 < V::rank) ? smr.sizes[1] : 0U,
    (1 < V::rank) ? smr.strides[1] : 0U,
    (2 < V::rank) ? smr.sizes[2] : 0U,
    (2 < V::rank) ? smr.strides[2] : 0U,
    (3 < V::rank) ? smr.sizes[3] : 0U,
    (3 < V::rank) ? smr.strides[3] : 0U,
    (4 < V::rank) ? smr.sizes[4] : 0U,
    (4 < V::rank) ? smr.strides[4] : 0U,
    (5 < V::rank) ? smr.sizes[5] : 0U,
    (5 < V::rank) ? smr.strides[5] : 0U,
    (6 < V::rank) ? smr.sizes[6] : 0U,
    (6 < V::rank) ? smr.strides[6] : 0U,
    (7 < V::rank) ? smr.sizes[7] : 0U,
    (7 < V::rank) ? smr.strides[7] : 0U);
    return V(&smr.data[smr.offset], layout);
  }
  Layout layout(
    (0 < V::rank) ? smr.sizes[0] : 0U,
    (1 < V::rank) ? smr.sizes[1] : 0U,
    (2 < V::rank) ? smr.sizes[2] : 0U,
    (3 < V::rank) ? smr.sizes[3] : 0U,
    (4 < V::rank) ? smr.sizes[4] : 0U,
    (5 < V::rank) ? smr.sizes[5] : 0U,
    (6 < V::rank) ? smr.sizes[6] : 0U,
    (7 < V::rank) ? smr.sizes[7] : 0U);
  if constexpr(std::is_same_v<Layout, Kokkos::LayoutLeft>)
  {
    int64_t expectedStride = 1;
    for(int i = 0; i < V::rank; i++)
    {
      if(expectedStride != smr.strides[i])
        Kokkos::abort("Cannot convert non-contiguous StridedMemRefType to LayoutLeft Kokkos::View");
      expectedStride *= smr.sizes[i];
    }
  }
  else if constexpr(std::is_same_v<Layout, Kokkos::LayoutRight>)
  {
    int64_t expectedStride = 1;
    for(int i = V::rank - 1; i >= 0; i--)
    {
      if(expectedStride != smr.strides[i])
        Kokkos::abort("Cannot convert non-contiguous StridedMemRefType to LayoutRight Kokkos::View");
      expectedStride *= smr.sizes[i];
    }
  }
  return V(&smr.data[smr.offset], layout);
}

// builtin.module
// func.func
#ifndef PYTACO_CPP_DRIVER
extern "C" void _mlir_ciface_sparseValuesF32(StridedMemRefType<float, 1>*, int8_t*);
#endif

// func.func
#ifndef PYTACO_CPP_DRIVER
extern "C" void _mlir_ciface_sparseIndices0(StridedMemRefType<size_t, 1>*, int8_t*, size_t);
#endif

// func.func
#ifndef PYTACO_CPP_DRIVER
extern "C" void _mlir_ciface_sparsePointers0(StridedMemRefType<size_t, 1>*, int8_t*, size_t);
#endif

// func.func
#ifndef PYTACO_CPP_DRIVER
extern "C" int8_t* _mlir_ciface_newSparseTensor(StridedMemRefType<size_t, 1>*, StridedMemRefType<size_t, 1>*, StridedMemRefType<int8_t, 1>*, StridedMemRefType<size_t, 1>*, StridedMemRefType<size_t, 1>*, int32_t, int32_t, int32_t, int32_t, int8_t*);
#endif

// func.func
int8_t* pytaco_main(int8_t* v1, int8_t* v2) {
  std::cout << "Hello from MLIR-Kokkos function pytaco_main!\n";
  // memref.alloca
  Kokkos::View<int8_t[1], Kokkos::LayoutRight> v3 = Kokkos::View<int8_t[1], Kokkos::LayoutRight>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v3"));
  // memref.cast
  Kokkos::View<int8_t*, Kokkos::LayoutRight> v4(v3.data(), 1);
  // memref.store
  v3(0) = 4;
  // memref.alloca
  Kokkos::View<size_t[1], Kokkos::LayoutRight> v5 = Kokkos::View<size_t[1], Kokkos::LayoutRight>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v5"));
  // memref.cast
  Kokkos::View<size_t*, Kokkos::LayoutRight> v6(v5.data(), 1);
  // memref.store
  v5(0) = 5;
  // memref.alloca
  Kokkos::View<size_t[1], Kokkos::LayoutRight> v7 = Kokkos::View<size_t[1], Kokkos::LayoutRight>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v7"));
  // memref.cast
  Kokkos::View<size_t*, Kokkos::LayoutRight> v8(v7.data(), 1);
  // memref.store
  v7(0) = 5;
  // memref.alloca
  Kokkos::View<size_t[1], Kokkos::LayoutRight> v9 = Kokkos::View<size_t[1], Kokkos::LayoutRight>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v9"));
  // memref.cast
  Kokkos::View<size_t*, Kokkos::LayoutRight> v10(v9.data(), 1);
  // memref.store
  v9(0) = 0;
  // memref.alloca
  Kokkos::View<size_t[1], Kokkos::LayoutRight> v11 = Kokkos::View<size_t[1], Kokkos::LayoutRight>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v11"));
  // memref.cast
  Kokkos::View<size_t*, Kokkos::LayoutRight> v12(v11.data(), 1);
  // memref.store
  v11(0) = 0;
  // llvm.mlir.null
  int8_t* v13 = nullptr;
  // func.call
  int8_t* v14;
  {
    StridedMemRefType<size_t, 1> v6_smr = viewToStridedMemref(v6);
    StridedMemRefType<size_t, 1> v8_smr = viewToStridedMemref(v8);
    StridedMemRefType<int8_t, 1> v4_smr = viewToStridedMemref(v4);
    StridedMemRefType<size_t, 1> v10_smr = viewToStridedMemref(v10);
    StridedMemRefType<size_t, 1> v12_smr = viewToStridedMemref(v12);
    v14 = _mlir_ciface_newSparseTensor(&v6_smr, &v8_smr, &v4_smr, &v10_smr, &v12_smr, 0, 0, 2, 0, v13);
  }
  ;
  // func.call
  Kokkos::View<size_t*, Kokkos::LayoutRight> v15;
  {
    StridedMemRefType<size_t, 1> v15_smr;
    _mlir_ciface_sparsePointers0(&v15_smr, v1, 1);
    v15 = stridedMemrefToView<Kokkos::View<size_t*, Kokkos::LayoutRight>>(v15_smr);
  }
  ;
  // func.call
  Kokkos::View<size_t*, Kokkos::LayoutRight> v16;
  {
    StridedMemRefType<size_t, 1> v16_smr;
    _mlir_ciface_sparseIndices0(&v16_smr, v1, 1);
    v16 = stridedMemrefToView<Kokkos::View<size_t*, Kokkos::LayoutRight>>(v16_smr);
  }
  ;
  // func.call
  Kokkos::View<float*, Kokkos::LayoutRight> v17;
  {
    StridedMemRefType<float, 1> v17_smr;
    _mlir_ciface_sparseValuesF32(&v17_smr, v1);
    v17 = stridedMemrefToView<Kokkos::View<float*, Kokkos::LayoutRight>>(v17_smr);
  }
  ;
  // func.call
  Kokkos::View<float*, Kokkos::LayoutRight> v18;
  {
    StridedMemRefType<float, 1> v18_smr;
    _mlir_ciface_sparseValuesF32(&v18_smr, v2);
    v18 = stridedMemrefToView<Kokkos::View<float*, Kokkos::LayoutRight>>(v18_smr);
  }
  ;
  // func.call
  Kokkos::View<float*, Kokkos::LayoutRight> v19;
  {
    StridedMemRefType<float, 1> v19_smr;
    _mlir_ciface_sparseValuesF32(&v19_smr, v14);
    v19 = stridedMemrefToView<Kokkos::View<float*, Kokkos::LayoutRight>>(v19_smr);
  }
  ;
  // scf.parallel
  Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0, (5 - 0 + 1 - 1) / 1),
  KOKKOS_LAMBDA(int64_t unit_v20)
  {
    int64_t v20 = 0 + unit_v20 * 1;
    // memref.load
    float v21 = v19(v20);
    // memref.load
    size_t v22 = v15(v20);
    // arith.addi
    size_t v23 = v20 + 1;
    // memref.load
    size_t v24 = v15(v23);
    // scf.for
    float v25;
    float v26 = v21;
    for (size_t v27 = v22; v27 < v24; v27 += 1) {
      // memref.load
      size_t v28 = v16(v27);
      // memref.load
      float v29 = v17(v27);
      // memref.load
      float v30 = v18(v28);
      // arith.mulf
      float v31 = v29 * v30;
      // arith.addf
      float v32 = v26 + v31;
      v26 = v32;
    }
    v25 = v26;;
    // memref.store
    v19(v20) = v25;
    // scf.yield
    ;
  })
  ;
  // func.return
  return v14;
}

extern "C" void py_pytaco_main(int8_t** ret0, int8_t* param0, int8_t* param1)
{
  std::cout << "Starting MLIR function on process " << getpid() << '\n';
  std::cout << "Optionally attach debugger now, then press <Enter> to continue: ";
  std::cin.get();
  auto results = pytaco_main(param0, param1);
  *ret0 = results;
}


extern "C" void lapis_initialize()
{
  Kokkos::initialize();
}

extern "C" void lapis_finalize()
{
  Kokkos::finalize();
}
