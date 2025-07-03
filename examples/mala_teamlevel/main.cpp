#include "forward_snap.hpp"
#include <fstream>
#include <iostream>

using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using TeamMem = typename TeamPol::member_type;
using Float2D = Kokkos::View<float**, Kokkos::DefaultExecutionSpace>;

void run(const Float2D& predictions, const Float2D& descriptors) {
  // Allow up to this much level 0 scratch (aka shared mem) per team
  // when deciding which temporary views to spill into level 1 scratch (global pool).
  constexpr int max_shared = 4096;
  // This is a number that fits within all modern 
  // Determine L0 and L1 per-team scratch size
  constexpr int scratch0_required = forward_L0_scratch_required(max_shared);
  constexpr int scratch1_required = forward_L1_scratch_required(max_shared);
  std::cout << "With L0 limited to " << max_shared << " bytes, kernel requires " << scratch0_required << " bytes L0 and " << scratch1_required << " bytes L1\n";
  std::cout << "Running over " << descriptors.extent(0) << " descriptors.\n";
  GlobalViews_forward globalViews;
  TeamPol policy(descriptors.extent(0), Kokkos::AUTO());
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch0_required));
  policy.set_scratch_size(1, Kokkos::PerTeam(scratch1_required));
  // Execute the appropriate specialization based on shared amount
  Kokkos::parallel_for(policy,
    KOKKOS_LAMBDA(const TeamMem& t) {
      // Get 1D subviews for just one descriptor/prediction instance.
      auto pred = Kokkos::subview(predictions, t.league_rank(), Kokkos::ALL());
      auto desc = Kokkos::subview(descriptors, t.league_rank(), Kokkos::ALL());
      char* scratch0 = (char*)(t.team_scratch(0).get_shmem(scratch0_required));
      char* scratch1 = (char*)(t.team_scratch(1).get_shmem(scratch1_required));
      forward<Kokkos::DefaultExecutionSpace, max_shared>(t, globalViews, pred, desc, scratch0, scratch1);
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
    // Number of descriptors in snap_descriptors.txt (but could be anything in a real application)
    const int num_descriptors = 8748;
    // Dimension of input and output vectors for each descriptor (fixed by model)
    const int in_dimension = 91;
    const int out_dimension = 11;

    Kokkos::View<float**, Kokkos::DefaultExecutionSpace> descriptors("descriptors", num_descriptors, in_dimension);
    auto descriptors_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), descriptors);
    Kokkos::View<float**, Kokkos::DefaultExecutionSpace> predictions("predictions", num_descriptors, out_dimension);
    // Construct inputs
    {
      std::ifstream f("../data/snap_descriptors.txt");
      if(!f.good())
        throw std::runtime_error("Failed to open descriptor file");
      for(int i = 0; i < 8748; i++) {
        for(int j = 0; j < 91; j++) {
          f >> descriptors_host(i, j);
        }
      }
      f.close();
    }
    Kokkos::deep_copy(descriptors, descriptors_host);
    run(predictions, descriptors);
    auto predictions_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), predictions);
    std::cout << "Checking results...\n";
    Kokkos::View<float[8748][11], Kokkos::HostSpace> goldResults("goldResults");
    std::ifstream f("../data/snap_predictions.txt");
    if(!f.good())
      throw std::runtime_error("Failed to open predictions file");
    for(int i = 0; i < 8748; i++) {
      for(int j = 0; j < 11; j++) {
        f >> goldResults(i, j);
      }
    }
    f.close();
    float maxDiff = 0;
    for(int i = 0; i < 8748; i++) {
      for(int j = 0; j < 11; j++) {
        float diff = Kokkos::fabs(goldResults(i, j) - predictions_host(i, j));
        if(diff > maxDiff)
          maxDiff = diff;
      }
    }
    std::cout << "Maximum elementwise diff: " << maxDiff << '\n';
    passed = maxDiff < 1e-5;
  }
  lapis_finalize();
  Kokkos::finalize();
  if(passed)
    std::cout << "** Sucess**\n";
  else
    std::cout << "** Failure **\n";
  return passed ? 0 : 1;
}

