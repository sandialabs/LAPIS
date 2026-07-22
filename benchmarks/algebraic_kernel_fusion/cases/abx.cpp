#include LAPIS_BENCHMARK_MODULE

#include "../BenchmarkSupport.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

#ifndef LAPIS_ABX_M
#define LAPIS_ABX_M 128
#define LAPIS_ABX_K 256
#define LAPIS_ABX_N 1024
#define LAPIS_ABX_CASE "abx"
#endif

constexpr std::size_t m = LAPIS_ABX_M;
constexpr std::size_t k = LAPIS_ABX_K;
constexpr std::size_t n = LAPIS_ABX_N;

using MatrixA = LAPIS::DualView<float[m][k], Kokkos::LayoutRight>;
using MatrixB = LAPIS::DualView<float[k][n], Kokkos::LayoutRight>;
using VectorX = LAPIS::DualView<float[n], Kokkos::LayoutRight>;

void initialize(MatrixA &a, MatrixB &b, VectorX &x) {
  auto aHost = a.host_view();
  auto bHost = b.host_view();
  auto xHost = x.host_view();
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < k; ++column) {
      const int value = static_cast<int>((row * 17 + column * 13 + 3) % 19);
      aHost(row, column) = static_cast<float>(value - 9) * 0.015625F;
    }
  }
  for (std::size_t row = 0; row < k; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      const int value = static_cast<int>((row * 7 + column * 5 + 1) % 23);
      bHost(row, column) = static_cast<float>(value - 11) * 0.0078125F;
    }
  }
  for (std::size_t index = 0; index < n; ++index) {
    const int value = static_cast<int>((index * 11 + 5) % 29);
    xHost(index) = static_cast<float>(value - 14) * 0.03125F;
  }
  a.modifyHost();
  b.modifyHost();
  x.modifyHost();
}

double validateAndChecksum(MatrixA &a, MatrixB &b, VectorX &x) {
  auto result = abx(a, b, x);
  Kokkos::fence();
  result.syncHost();

  const auto aHost = a.host_view();
  const auto bHost = b.host_view();
  const auto xHost = x.host_view();
  const auto resultHost = result.host_view();

  double checksum = 0.0;
  for (std::size_t row = 0; row < m; ++row) {
    double expected = 0.0;
    for (std::size_t inner = 0; inner < k; ++inner) {
      double bx = 0.0;
      for (std::size_t column = 0; column < n; ++column)
        bx += static_cast<double>(bHost(inner, column)) * xHost(column);
      expected += static_cast<double>(aHost(row, inner)) * bx;
    }
    const double actual = resultHost(row);
    const double tolerance = 2.0e-3 + 2.0e-4 * std::abs(expected);
    if (std::abs(actual - expected) > tolerance) {
      throw std::runtime_error(
          "ABx validation failed at row " + std::to_string(row) + ": got " +
          std::to_string(actual) + ", expected " + std::to_string(expected));
    }
    checksum += actual;
  }
  return checksum;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = lapis::benchmark::parseOptions(argc, argv);
    lapis_initialize();
    {
      MatrixA a(std::string("A"));
      MatrixB b(std::string("B"));
      VectorX x(std::string("x"));
      initialize(a, b, x);

      const double checksum = validateAndChecksum(a, b, x);
      const auto measurement =
          lapis::benchmark::measure(options, [&] { return abx(a, b, x); });
      lapis::benchmark::printResult(LAPIS_ABX_CASE, LAPIS_BENCHMARK_VARIANT,
                                    options, measurement, checksum);
    }
    lapis_finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
