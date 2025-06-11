#include "forward_snap.hpp"
#include <fstream>

using TeamPol = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using TeamMem = typename TeamPol::member_type;
using Float2D = Kokkos::View<float**, Kokkos::DefaultExecutionSpace>;

// Note: signature of forward function will look like this.
// The example below lets layout0 and layout1 be inferred, so the calling
// KOKKOS_LAMBDA doesn't depend on the view layouts.
//
// template<int max_shared, typename layout0, typename layout1>
// KOKKOS_INLINE_FUNCTION void forward(
//  const TeamMem& T,
//  const Kokkos::View<float*, layout0, DefaultExecutionSpace>& pred,
//  const Kokkos::View<float*, layout1, DefaultExecutionSpace>& desc);

void run(const Float2D& predictions, const Float2D& descriptors) {
  // Allow up to this much level 0 scratch (aka shared mem) per team
  // when deciding which temporary views to spill into level 1 scratch (global pool).
  //
  // This is just an example; the MALA model in fact needs only 444 bytes per team.
  // So scratch0 is expected to be 444 and scratch1 should be 0.
  constexpr int max_shared = 1024;
  // This is a number that fits within all modern 
  // Determine L0 and L1 per-team scratch size
  int scratch0 = forward_scratch_0<max_shared>();
  int scratch1 = forward_scratch_1<max_shared>();
  Policy policy(descriptors.extent(0), Kokkos::AUTO());
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch0));
  policy.set_scratch_size(1, Kokkos::PerTeam(scratch1));
  // Execute the appropriate specialization based on shared amount
  Kokkos::parallel_for(policy,
    KOKKOS_LAMBDA(const TeamMem& t) {
      // Get 1D subviews for just one descriptor/prediction instance.
      auto pred = Kokkos::subview(predictions, t.league_rank(), Kokkos::ALL());
      auto desc = Kokkos::subview(descriptors, t.league_rank(), Kokkos::ALL());
      forward<max_shared>(t, pred, desc);
    });
}

int main()
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  lapis_initialize();
  {
    ExecSpace().print_configuration(std::cout);
    Kokkos::Timer t;
    int numTrials = 10000;
    // Number of descriptors in snap_descriptors.txt (but could be anything in a real application)
    const int num_descriptors = 8748;
    // Dimension of input and output vectors for each descriptor (fixed by model)
    const int in_dimension = 91;
    const int out_dimension = 11;

     descriptors("descriptors", num_descriptors, in_dimension);
    auto descriptors_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), descriptors);
    Kokkos::View<float**, Kokkos::DefaultExecutionSpace> predictions("predictions", num_descriptors, out_dimension);
    // Construct inputs
    {
      std::ifstream f("MALA/snap_descriptors.txt");
      for(int i = 0; i < 8748; i++) {
        for(int j = 0; j < 91; j++) {
          f >> descriptors_host(i, j);
        }
      }
      f.close();
    }
    Kokkos::deep_copy(descriptors, descriptors_host);
    // Warmup
    run(predictions, descriptors);
    Kokkos::fence();
    t.reset();
    for(int i = 0; i < numTrials; i++) {
      run(predictions, descriptors);
      Kokkos::fence();
    }
    double elapsed = t.seconds();
    std::cout << "MALA (team based) inference avg time = " << elapsed / numTrials << "\n";
  }
  lapis_finalize();
  return 0;
}

