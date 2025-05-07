#include "batched_gemm.hpp"
#include <Kokkos_Random.hpp>

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using DualV = LAPIS::DualView<float***, Kokkos::LayoutRight>;
  using DeviceV = typename DualV::DeviceView;
  using HostV = typename DualV::HostView;
  using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
  lapis_initialize();
  {
    ExecSpace().print_configuration(std::cout);
    RandPool pool(123);
    Kokkos::Timer t;
    int numTrials = 100;
    // Number of matrices to multiply
    int b = 1 << 20;
    // Multiply matrices of size 2*2 up to nmax*nmax
    int nmax = 32;
    for(int n = 2; n <= nmax; n += 2) {
      // Construct inputs
      DualV A("A", b, n, n);
      DualV B("B", b, n, n);
      DualV C("C", b, n, n);
      A.modifiedDevice();
      Kokkos::fill_random(A.device_view(), pool, 0.0, 1.0);
      B.modifiedDevice();
      Kokkos::fill_random(B.device_view(), pool, 0.0, 1.0);
      A.syncHost();
      B.syncHost();
      // Untimed warmup
      forward(A, B, C);
      Kokkos::fence();
      t.reset();
      for(int i = 0; i < numTrials; i++) {
        forward(A, B, C);
        Kokkos::fence();
      }
      double elapsed = t.seconds();
      std::cout << "Block size = " << n << ": avg time = " << elapsed / numTrials << "\n";
    }
  }
  lapis_finalize();
  return 0;
}
