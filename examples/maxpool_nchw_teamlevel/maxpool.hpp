#ifndef LAPIS_MODULE_H
#define LAPIS_MODULE_H
#include <Kokkos_Core.hpp>
// Return the total amount of scratch that the function uses
// This is the upper bound to what forward_L0_scratch_required can return
KOKKOS_INLINE_FUNCTION constexpr int forward_total_scratch_required() {
  return 8448;
}

// Find L1 address shift: for allocations spilling into level 1 scratch,
// this is subtracted from the allocation address to find its relative address within L1.
KOKKOS_INLINE_FUNCTION constexpr int forward_L1_shift(int L0_scratch_max) {
  int tmp = 8448;
  if(8448 > L0_scratch_max && 0 < tmp)
    tmp = 0;
  return tmp;
}

// Find the actual level 0 scratch required by the function, assuming this limit.
// The answer will be at most L0_scratch_max but never larger.
KOKKOS_INLINE_FUNCTION constexpr int forward_L0_scratch_required(int L0_scratch_max) {
  int tmp = 0;
  if(8448 <= L0_scratch_max && 8448 > tmp)
    tmp = 8448;
  return tmp;
}

// Find the actual level 1 scratch required by the function, assuming this limit for L0 scratch.
// This has no strict upper bound.
KOKKOS_INLINE_FUNCTION constexpr int forward_L1_scratch_required(int L0_scratch_max) {
  return 8448 - forward_L1_shift(L0_scratch_max);
}

template<typename ExecSpace, int L0_scratch_max, typename ViewArg0, typename ViewArg1>
KOKKOS_INLINE_FUNCTION void forward(const typename Kokkos::TeamPolicy<ExecSpace>::member_type& team, const ViewArg0& v1, const ViewArg1& v2, char* scratch0, char* scratch1) {
  constexpr int l1_cutoff = forward_L1_shift(L0_scratch_max);
  constexpr bool v3_spill = 8448 > L0_scratch_max;
  Kokkos::View<float[1][3][22][32], Kokkos::LayoutRight, Kokkos::AnonymousSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v3((float*) (v3_spill ? (scratch1 + 0 - l1_cutoff) : (scratch0 + 0)));
  ;
  Kokkos::parallel_for(Kokkos::TeamVectorMDRange(team, 1, 3, 22, 32),
  [=](size_t v4, size_t v5, size_t v6, size_t v7) {
    v3(v4, v5, v6, v7) = -INFINITY;
  });
  team.team_barrier();
  Kokkos::LayoutStride v8_layout(1, 2112, 3, 704, 20, 32, 30, 1);
  Kokkos::View<float[1][3][20][30], Kokkos::LayoutStride, Kokkos::AnonymousSpace> v8(v3.data() + 33, v8_layout);
  ;
  Kokkos::parallel_for(Kokkos::TeamVectorMDRange(team, 1, 3, 20, 30),
  [=](size_t v9, size_t v10, size_t v11, size_t v12) {
    float v13 = v2(v9, v10, v11, v12);
    v8(v9, v10, v11, v12) = v13;
  });
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorMDRange(team, 1, 3, 6, 10),
  [=](size_t v14, size_t v15, size_t v16, size_t v17) {
    v1(v14, v15, v16, v17) = -INFINITY;
  });
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorMDRange(team, 1, 3, 6, 10),
  [=](size_t v18, size_t v19, size_t v20, size_t v21) {
    for (size_t v22 = 0; v22 < 3; v22 += 1) {
      for (size_t v23 = 0; v23 < 3; v23 += 1) {
        size_t v24 = v20 * 3;
        size_t v25 = v22 * 2;
        size_t v26 = v24 + v25;
        size_t v27 = v21 * 3;
        size_t v28 = v23 * 2;
        size_t v29 = v27 + v28;
        float v30 = v3(v18, v19, v26, v29);
        float v31 = v1(v18, v19, v20, v21);
        float v32 = (v31 > v30) ? v31 : v30;
        v1(v18, v19, v20, v21) = v32;
      };
    };
  });
  team.team_barrier();
  return;
}



extern "C" void lapis_initialize();
extern "C" void lapis_finalize();
#endif
