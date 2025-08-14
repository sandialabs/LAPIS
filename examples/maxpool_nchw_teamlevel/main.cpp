#include "maxpool.hpp"
#include <fstream>
#include <iostream>

using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using TeamMem = typename TeamPol::member_type;
using Float4D = Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using Float4D_Host = Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>;

void run(const Float4D& out, const Float4D& in) {
  // Allow up to this much level 0 scratch (aka shared mem) per team
  // when deciding which temporary views to spill into level 1 scratch (global pool).
  constexpr int max_shared = 12288;
  // This is a number that fits within all modern 
  // Determine L0 and L1 per-team scratch size
  constexpr int scratch0_required = forward_L0_scratch_required(max_shared);
  constexpr int scratch1_required = forward_L1_scratch_required(max_shared);
  std::cout << "With L0 limited to " << max_shared << " bytes, kernel requires " << scratch0_required << " bytes L0 and " << scratch1_required << " bytes L1\n";
  std::cout << "Running over " << in.extent(0) << " images.\n";
  TeamPol policy(in.extent(0), Kokkos::AUTO());
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch0_required));
  policy.set_scratch_size(1, Kokkos::PerTeam(scratch1_required));
  // Execute the appropriate specialization based on shared amount
  Kokkos::parallel_for(policy,
    KOKKOS_LAMBDA(const TeamMem& t) {
      // Get 1D subviews for just one descriptor/prediction instance.
      int i = t.league_rank();
      auto in_sub = Kokkos::subview(in, Kokkos::make_pair(i, i+1), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      auto out_sub = Kokkos::subview(out, Kokkos::make_pair(i, i+1), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      char* scratch0 = (char*)(t.team_scratch(0).get_shmem(scratch0_required));
      char* scratch1 = (char*)(t.team_scratch(1).get_shmem(scratch1_required));
      forward<Kokkos::DefaultExecutionSpace, max_shared>(t, out_sub, in_sub, scratch0, scratch1);
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
    Float4D_Host in_host("maxpool input", 2, 3, 20, 30);
    Float4D_Host out_gold("maxpool output", 2, 3, 6, 10);
    Float4D in("maxpool input", 2, 3, 20, 30);
    Float4D out("maxpool output", 2, 3, 6, 10);
    // Construct inputs
    {
      std::ifstream f("../data/maxpool_input.txt");
      if(!f.good())
        throw std::runtime_error("Failed to open ../data/maxpool_input.txt");
      for(size_t i = 0; i < in_host.size(); i++) {
        f >> in_host.data()[i];
      }
      f.close();
    }
    // Read correct output values
    {
      std::ifstream f("../data/maxpool_output.txt");
      if(!f.good())
        throw std::runtime_error("Failed to open ../data/maxpool_output.txt");
      for(size_t i = 0; i < out_gold.size(); i++) {
        f >> out_gold.data()[i];
      }
      f.close();
    }
    Kokkos::deep_copy(in, in_host);
    run(out, in);
    auto out_actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    std::cout << "Checking results...\n";
    float maxDiff = 0;
    for(int i = 0; i < 360; i++) {
      float diff = Kokkos::fabs(out_gold.data()[i] - out_actual.data()[i]);
      if(diff > maxDiff)
        maxDiff = diff;
    }
    std::cout << "Maximum elementwise diff: " << maxDiff << '\n';
    passed = maxDiff < 1e-5;
  }
  lapis_finalize();
  Kokkos::finalize();
  if(passed)
    std::cout << "** Success **\n";
  else
    std::cout << "** Failure **\n";
  return passed ? 0 : 1;
}

