#include "forward_snap.hpp"
#include <fstream>
#include <iostream>

// Number of descriptors in snap_descriptors.txt (but could be anything in a real application)
constexpr int num_descriptors = 8748;
// Dimension of input and output vectors for each descriptor (fixed by model)
constexpr int in_dimension = 91;
constexpr int out_dimension = 11;

using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using TeamMem = typename TeamPol::member_type;
using DualV = LAPIS::DualView<float[num_descriptors][in_dimension], Kokkos::LayoutRight>;

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  // Note: lapis_initialize does not initialize Kokkos for team-level kernels
  lapis_initialize();
  std::cout << "Running on " << ExecSpace().name() << " backend.\n";
  bool passed = true;
  {
    DualV descriptors(std::string("descriptors"));
    // Construct inputs
    {
      std::ifstream f("../data/snap_descriptors.txt");
      if(!f.good())
        throw std::runtime_error("Failed to open descriptor file");
      for(int i = 0; i < num_descriptors; i++) {
        for(int j = 0; j < in_dimension; j++) {
          f >> descriptors.host_view()(i, j);
        }
      }
      f.close();
    }
    descriptors.modifyHost();
    auto predictions = forward(descriptors);
    predictions.syncHost();
    auto predictions_host = predictions.host_view();
    std::cout << "Checking results...\n";
    Kokkos::View<float[num_descriptors][out_dimension], Kokkos::HostSpace> goldResults("goldResults");
    std::ifstream f("../data/snap_predictions.txt");
    if(!f.good())
      throw std::runtime_error("Failed to open predictions file");
    for(int i = 0; i < num_descriptors; i++) {
      for(int j = 0; j < out_dimension; j++) {
        f >> goldResults(i, j);
      }
    }
    f.close();
    float maxDiff = 0;
    for(int i = 0; i < num_descriptors; i++) {
      for(int j = 0; j < out_dimension; j++) {
        float diff = Kokkos::fabs(goldResults(i, j) - predictions_host(i, j));
        if(diff > maxDiff)
          maxDiff = diff;
      }
    }
    std::cout << "Maximum elementwise diff: " << maxDiff << '\n';
    passed = maxDiff < 1e-5;
  }
  lapis_finalize();
  if(passed)
    std::cout << "** Success **\n";
  else
    std::cout << "** Failure **\n";
  return passed ? 0 : 1;
}

