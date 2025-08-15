#include "cloudfrac.hpp"
#include <fstream>
#include <iostream>

using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using TeamMem = typename TeamPol::member_type;
using Float2D = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using Float2D_Host = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>;

void read2D(const Float2D_Host& tensor, std::string path)
{
  std::ifstream f(path);
  if(!f.good())
    throw std::runtime_error("Failed to open " + path);
  for(size_t i = 0; i < tensor.size(); i++) {
    f >> tensor.data()[i];
  }
  f.close();
}

void run(const Float2D& out1, const Float2D& out2, const Float2D& in1, const Float2D& in2) {
  // Allow up to this much level 0 scratch (aka shared mem) per team
  // when deciding which temporary views to spill into level 1 scratch (global pool).
  constexpr int max_shared = 8192;
  // This is a number that fits within all modern 
  // Determine L0 and L1 per-team scratch size
  constexpr int scratch0_required = forward_L0_scratch_required(max_shared);
  constexpr int scratch1_required = forward_L1_scratch_required(max_shared);
  std::cout << "With L0 limited to " << max_shared << " bytes, kernel requires " << scratch0_required << " bytes L0 and " << scratch1_required << " bytes L1\n";
  std::cout << "Running over " << in1.extent(0) << " descriptors.\n";
  TeamPol policy(in1.extent(0), Kokkos::AUTO());
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch0_required));
  policy.set_scratch_size(1, Kokkos::PerTeam(scratch1_required));
  GlobalViews_forward globalViews;
  // Execute the appropriate specialization based on shared amount
  Kokkos::parallel_for(policy,
    KOKKOS_LAMBDA(const TeamMem& t) {
      // Get 1D subviews for just one descriptor/prediction instance.
      int i = t.league_rank();
      auto out1_sub = Kokkos::subview(out1, i, Kokkos::ALL());
      auto out2_sub = Kokkos::subview(out2, i, Kokkos::ALL());
      auto in1_sub = Kokkos::subview(in1, i, Kokkos::ALL());
      auto in2_sub = Kokkos::subview(in2, i, Kokkos::ALL());
      char* scratch0 = (char*)(t.team_scratch(0).get_shmem(scratch0_required));
      char* scratch1 = (char*)(t.team_scratch(1).get_shmem(scratch1_required));
      forward<Kokkos::DefaultExecutionSpace, max_shared>(t, globalViews, out1_sub, out2_sub, in1_sub, in2_sub, scratch0, scratch1);
    });
}

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  Kokkos::initialize();
  std::cout << "Running on " << ExecSpace().name() << " backend.\n";
  // Note: lapis_initialize does not initialize Kokkos for team-level kernels
  lapis_initialize();
  bool passed = true;
  {
    int b = 16;
    Float2D_Host in1_host("in1", b, 72);
    Float2D_Host in2_host("in2", b, 72);
    Float2D_Host out1_gold("out1 gold", b, 72);
    Float2D_Host out2_gold("out2 gold", b, 72);
    read2D(in1_host, "../data/cloudfrac_in1.txt");
    read2D(in2_host, "../data/cloudfrac_in2.txt");
    read2D(out1_gold, "../data/cloudfrac_out1.txt");
    read2D(out2_gold, "../data/cloudfrac_out2.txt");
    Float2D in1("in1", b, 72);
    Float2D in2("in2", b, 72);
    Float2D out1("out1", b, 72);
    Float2D out2("out2", b, 72);
    Kokkos::deep_copy(in1, in1_host);
    Kokkos::deep_copy(in2, in2_host);
    run(out1, out2, in1, in2);
    auto out1_actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out1);
    auto out2_actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out2);
    std::cout << "Checking results...\n";
    float maxDiff = 0;
    for(size_t i = 0; i < out1_gold.size(); i++) {
      float diff = Kokkos::fabs(out1_gold.data()[i] - out1_actual.data()[i]);
      if(diff > maxDiff)
        maxDiff = diff;
    }
    for(size_t i = 0; i < out2_gold.size(); i++) {
      float diff = Kokkos::fabs(out2_gold.data()[i] - out2_actual.data()[i]);
      if(diff > maxDiff)
        maxDiff = diff;
    }
    std::cout << "Maximum elementwise diff: " << maxDiff << '\n';
    passed = maxDiff < 1e-4;
  }
  lapis_finalize();
  Kokkos::finalize();
  if(passed)
    std::cout << "** Success **\n";
  else
    std::cout << "** Failure **\n";
  return passed ? 0 : 1;
}

