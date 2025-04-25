#include "forward_snap.hpp"
#include <fstream>

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using DualV = LAPIS::DualView<float[8748][91], Kokkos::LayoutRight>;
  lapis_initialize();
  {
    ExecSpace().print_configuration(std::cout);
    Kokkos::Timer t;
    int numTrials = 10000;
    // Construct inputs
    DualV input(std::string("in_descriptors"));
    {
      input.modifiedHost();
      std::ifstream f("MALA/snap_descriptors.txt");
      for(int i = 0; i < 8748; i++) {
        for(int j = 0; j < 91; j++) {
          f >> input.host_view()(i, j);
        }
      }
      f.close();
    }
    // Warmup
    auto output = forward(input);
    Kokkos::fence();
    t.reset();
    for(int i = 0; i < numTrials; i++) {
      output = forward(input);
      Kokkos::fence();
    }
    double elapsed = t.seconds();
    std::cout << "MALA inference avg time = " << elapsed / numTrials << "\n";
  }
  lapis_finalize();
  return 0;
}

