#include "spmv.hpp"
/*
LAPIS::DualView<double*, Kokkos::LayoutRight>
spmv(
  LAPIS::DualView<int32_t*, Kokkos::LayoutRight> v1, // A rowptrs
  LAPIS::DualView<int32_t*, Kokkos::LayoutRight> v2, // A entries
  LAPIS::DualView<double*, Kokkos::LayoutRight> v3,  // A values
  Struct0 v4,                                        // {{m, n}, {m+1, nnz, nnz}}
  LAPIS::DualView<double*, Kokkos::LayoutRight> v5,  // x
  LAPIS::DualView<double*, Kokkos::LayoutRight> v6); // y
*/
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include <vector>
#include <string>

using std::vector;
using std::string;

int main(int argc, const char** argv) {
  lapis_initialize();
  {
    Kokkos::Timer t;
    vector<string> matrices = {"Laplace3D_100", "Elasticity3D_60", "af_shell7"};
    using Int_DV = LAPIS::DualView<int32_t*, Kokkos::LayoutRight>;
    using Vector_DV = LAPIS::DualView<double*, Kokkos::LayoutRight>;
    if(argc != 2) {
      std::cout << "Expected one argument (matrix file in MatrixMarket format)\n";
      return 1;
    }
    std::string mfile = argv[1];
    // Construct inputs
    using Scalar = double;
    using Offset = int32_t;
    using Ordinal = int32_t;
    std::cout << "Reading matrix from " << mfile << "...\n";
    auto A = KokkosSparse::read_kokkos_crst_matrix<
      KokkosSparse::CrsMatrix<Scalar, Ordinal, Kokkos::DefaultExecutionSpace, void, Offset>>(mfile.c_str());
    std::cout << "Read matrix (" << A.numRows() << "x" << A.numCols()
      << " with " << A.nnz() << " entries)\n";
    // Have to get rid of const on rowptrs, which StaticCrsGraph always adds
    typename Int_DV::DeviceView A_rowptrs_device(const_cast<int32_t*>(A.graph.row_map.data()), A.graph.row_map.extent(0));
    Int_DV    A_rowptrs(A_rowptrs_device);
    Int_DV    A_entries(A.graph.entries);
    Vector_DV A_values(A.values);
    Vector_DV x("x", A.numCols());
    Vector_DV y("y", A.numRows());
    Struct0 shape = {
      {A.numRows(), A.numCols()},
      {A_rowptrs.extent(0), A_entries.extent(0), A_values.extent(0)}
    };
    // Warmup
    spmv(A_rowptrs, A_entries, A_values, shape, x, y);
    Kokkos::fence();
    const int numTrials = 1000;
    t.reset();
    for(int i = 0; i < numTrials; i++) {
      spmv(A_rowptrs, A_entries, A_values, shape, x, y);
      Kokkos::fence();
    }
    double elapsed = t.seconds();
    std::cout << "spmv avg time = " << elapsed / numTrials << "\n";
  }
  lapis_finalize();
  return 0;
}
