#include "gemm.hpp"
#include <Kokkos_Random.hpp>

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using DualV = LAPIS::DualView<float**, Kokkos::LayoutRight>;
  using DeviceV = typename DualV::DeviceView;
  using HostV = typename DualV::HostView;
  using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
  lapis_initialize();
  {
    ExecSpace().print_configuration(std::cout);
    RandPool pool(123);
    Kokkos::Timer t;
    int numTrials = 1000;
    for(int n = 256; n <= 2048; n += 32) {
      // Construct inputs
      DualV A("A", n, n);
      DualV B("B", n, n);
      DualV C("C", n, n);
      A.modifyDevice();
      Kokkos::fill_random(A.device_view(), pool, 0.0, 1.0);
      B.modifyDevice();
      Kokkos::fill_random(B.device_view(), pool, 0.0, 1.0);
      A.syncHost();
      B.syncHost();
      // Warmup
      forward(A, B, C);
      Kokkos::fence();
      t.reset();
      for(int i = 0; i < numTrials; i++) {
        forward(A, B, C);
        Kokkos::fence();
      }
      double elapsed = t.seconds();
      std::cout << "Square matrix with n = " << n << ": avg time = " << elapsed / numTrials << "\n";
    }
  }
  lapis_finalize();
  return 0;
}
