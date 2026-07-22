#include LAPIS_BENCHMARK_MODULE

#include "../BenchmarkSupport.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

constexpr std::size_t size = 512;

using Matrix = LAPIS::DualView<double[size][size], Kokkos::LayoutRight>;
using Vector = LAPIS::DualView<double[size], Kokkos::LayoutRight>;

void initialize(Matrix &a, Vector &p, Vector &r, Vector &z, Vector &x,
                Vector &diagonalInverse) {
  auto aHost = a.host_view();
  auto pHost = p.host_view();
  auto rHost = r.host_view();
  auto zHost = z.host_view();
  auto xHost = x.host_view();
  auto diagonalInverseHost = diagonalInverse.host_view();

  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column < size; ++column) {
      aHost(row, column) =
          row == column ? 4.0
                        : (row + 1 == column || column + 1 == row ? -1.0 : 0.0);
    }
    pHost(row) = 0.25 + static_cast<double>((row * 7 + 3) % 17) * 0.03125;
    rHost(row) = -0.5 + static_cast<double>((row * 11 + 5) % 23) * 0.0625;
    xHost(row) = static_cast<double>((row * 13 + 1) % 19) * 0.015625;
    diagonalInverseHost(row) = 0.25;
    zHost(row) = diagonalInverseHost(row) * rHost(row);
  }
  a.modifyHost();
  p.modifyHost();
  r.modifyHost();
  z.modifyHost();
  x.modifyHost();
  diagonalInverse.modifyHost();
}

double validateAndChecksum(Matrix &a, Vector &p, Vector &r, Vector &z,
                           Vector &x, Vector &diagonalInverse) {
  auto result = pcg_step(a, p, r, z, x, diagonalInverse);
  Kokkos::fence();
  std::apply([](auto &...value) { (value.syncHost(), ...); }, result);

  const auto aHost = a.host_view();
  const auto pHost = p.host_view();
  const auto rHost = r.host_view();
  const auto zHost = z.host_view();
  const auto xHost = x.host_view();
  const auto diagonalInverseHost = diagonalInverse.host_view();
  const auto xnextHost = std::get<0>(result).host_view();
  const auto rnextHost = std::get<1>(result).host_view();
  const auto pnextHost = std::get<2>(result).host_view();
  const auto znextHost = std::get<3>(result).host_view();
  const auto apHost = std::get<4>(result).host_view();
  const auto papHost = std::get<5>(result).host_view();
  const auto rznextHost = std::get<6>(result).host_view();

  std::array<double, size> ap{};
  double rz = 0.0;
  for (std::size_t row = 0; row < size; ++row) {
    rz += rHost(row) * zHost(row);
    for (std::size_t column = 0; column < size; ++column)
      ap[row] += aHost(row, column) * pHost(column);
  }
  double pap = 0.0;
  for (std::size_t row = 0; row < size; ++row)
    pap += pHost(row) * ap[row];
  const double alpha = rz / pap;

  std::array<double, size> xnext{};
  std::array<double, size> rnext{};
  std::array<double, size> znext{};
  double rznext = 0.0;
  for (std::size_t row = 0; row < size; ++row) {
    xnext[row] = xHost(row) + alpha * pHost(row);
    rnext[row] = rHost(row) - alpha * ap[row];
    znext[row] = rnext[row] * diagonalInverseHost(row);
    rznext += rnext[row] * znext[row];
  }
  const double beta = rznext / rz;

  double checksum = papHost() + rznextHost();
  auto check = [](double actual, double expected, std::string_view name,
                  std::size_t index) {
    const double tolerance = 1.0e-9 + 1.0e-10 * std::abs(expected);
    if (std::abs(actual - expected) > tolerance) {
      throw std::runtime_error(std::string(name) +
                               " validation failed at index " +
                               std::to_string(index));
    }
  };
  check(papHost(), pap, "pAp", 0);
  check(rznextHost(), rznext, "rnext-znext", 0);
  for (std::size_t row = 0; row < size; ++row) {
    const double pnext = znext[row] + beta * pHost(row);
    check(xnextHost(row), xnext[row], "xnext", row);
    check(rnextHost(row), rnext[row], "rnext", row);
    check(pnextHost(row), pnext, "pnext", row);
    check(znextHost(row), znext[row], "znext", row);
    check(apHost(row), ap[row], "Ap", row);
    checksum += xnextHost(row) + rnextHost(row) + pnextHost(row) +
                znextHost(row) + apHost(row);
  }
  return checksum;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = lapis::benchmark::parseOptions(argc, argv);
    lapis_initialize();
    {
      Matrix a(std::string("A"));
      Vector p(std::string("p"));
      Vector r(std::string("r"));
      Vector z(std::string("z"));
      Vector x(std::string("x"));
      Vector diagonalInverse(std::string("diagonal inverse"));
      initialize(a, p, r, z, x, diagonalInverse);

      const double checksum =
          validateAndChecksum(a, p, r, z, x, diagonalInverse);
      const auto measurement = lapis::benchmark::measure(
          options, [&] { return pcg_step(a, p, r, z, x, diagonalInverse); });
      lapis::benchmark::printResult("pcg", LAPIS_BENCHMARK_VARIANT, options,
                                    measurement, checksum);
    }
    lapis_finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
