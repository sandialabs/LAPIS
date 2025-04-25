#include "resnet.hpp"
#include <fstream>

//LAPIS::DualView<float**, Kokkos::LayoutRight> forward(LAPIS::DualView<float****, Kokkos::LayoutRight> v1) {
int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using DualV = LAPIS::DualView<float****, Kokkos::LayoutRight>;
  lapis_initialize();
  {
    ExecSpace().print_configuration(std::cout);
    Kokkos::Timer t;
    int numTrials = 100;
    const int batchSize = 8;
    // Construct inputs
    DualV input("images", batchSize, 3, 224, 224);
    {
      input.modifiedHost();
      std::cout << "Reading input image and copying across batch...";
      std::ifstream f("resnet/dog_preprocessed.txt");
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 224; j++) {
          for(int k = 0; k < 224; k++) {
            float val;
            f >> val;
            for(int b = 0; b < batchSize; b++) {
              input.host_view()(b, i, j, k) = val;
            }
          }
        }
      }
      f.close();
      std::cout << "Done\n";
    }
    // Warmup
    t.reset();
    auto output = forward(input);
    Kokkos::fence();
    std::cout << "Warmup took " << t.seconds() << "\n";
    t.reset();
    for(int i = 0; i < numTrials; i++) {
      output = forward(input);
      Kokkos::fence();
    }
    double elapsed = t.seconds();
    std::cout << "ResNet inference avg time (batch size = " << batchSize << ") = " << elapsed / numTrials << "\n";
  }
  lapis_finalize();
  return 0;
}
