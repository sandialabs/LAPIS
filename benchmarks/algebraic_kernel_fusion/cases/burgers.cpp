#include LAPIS_BENCHMARK_MODULE

#include "../BenchmarkSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t gridSize = 128;
constexpr std::size_t timeSteps = 20;
constexpr double inverseTwoDx = 64.5;
constexpr double inverseDxSquared = 16641.0;
constexpr double viscosity = 0.01;
constexpr double timeStep = 0.0005;
constexpr double pi = 3.141592653589793238462643383279502884;

using Field = LAPIS::DualView<double[gridSize][gridSize], Kokkos::LayoutRight>;
using HostField = std::vector<double>;

std::size_t offset(std::size_t row, std::size_t column) {
  return row * gridSize + column;
}

double valueOrBoundary(const HostField &field, std::ptrdiff_t row,
                       std::ptrdiff_t column) {
  if (row < 0 || column < 0 || row >= static_cast<std::ptrdiff_t>(gridSize) ||
      column >= static_cast<std::ptrdiff_t>(gridSize))
    return 0.0;
  return field[offset(static_cast<std::size_t>(row),
                      static_cast<std::size_t>(column))];
}

void eulerStage(const HostField &input, HostField &output) {
  for (std::size_t row = 0; row < gridSize; ++row) {
    for (std::size_t column = 0; column < gridSize; ++column) {
      const auto signedRow = static_cast<std::ptrdiff_t>(row);
      const auto signedColumn = static_cast<std::ptrdiff_t>(column);
      const double center = input[offset(row, column)];
      const double north = valueOrBoundary(input, signedRow - 1, signedColumn);
      const double south = valueOrBoundary(input, signedRow + 1, signedColumn);
      const double west = valueOrBoundary(input, signedRow, signedColumn - 1);
      const double east = valueOrBoundary(input, signedRow, signedColumn + 1);

      const double ux = (east - west) * inverseTwoDx;
      const double uy = (south - north) * inverseTwoDx;
      const double laplacian =
          (west + east + north + south - 4.0 * center) * inverseDxSquared;
      const double rhs = viscosity * laplacian - center * (ux + uy);
      output[offset(row, column)] = center + timeStep * rhs;
    }
  }
}

HostField referenceSolution(const HostField &initial) {
  HostField current = initial;
  HostField stage1(initial.size());
  HostField stage2(initial.size());
  for (std::size_t step = 0; step < timeSteps; ++step) {
    eulerStage(current, stage1);
    eulerStage(stage1, stage2);
    for (std::size_t index = 0; index < current.size(); ++index)
      current[index] = 0.5 * (current[index] + stage2[index]);
  }
  return current;
}

HostField initialize(Field &field) {
  auto host = field.host_view();
  HostField initial(gridSize * gridSize);
  const double dx = 1.0 / static_cast<double>(gridSize + 1);
  for (std::size_t row = 0; row < gridSize; ++row) {
    const double y = static_cast<double>(row + 1) * dx;
    for (std::size_t column = 0; column < gridSize; ++column) {
      const double x = static_cast<double>(column + 1) * dx;
      const double value = 0.5 * std::sin(pi * x) * std::sin(pi * y);
      host(row, column) = value;
      initial[offset(row, column)] = value;
    }
  }
  field.modifyHost();
  return initial;
}

double validateAndChecksum(Field &initial, const HostField &reference) {
  auto result = burgers(initial);
  Kokkos::fence();
  result.syncHost();
  const auto actual = result.host_view();

  double checksum = 0.0;
  double maximumError = 0.0;
  for (std::size_t row = 0; row < gridSize; ++row) {
    for (std::size_t column = 0; column < gridSize; ++column) {
      const double expected = reference[offset(row, column)];
      const double value = actual(row, column);
      maximumError = std::max(maximumError, std::abs(value - expected));
      checksum += value;
    }
  }
  if (maximumError > 1.0e-9)
    throw std::runtime_error("Burgers validation failed: maximum error " +
                             std::to_string(maximumError));
  return checksum;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = lapis::benchmark::parseOptions(argc, argv);
    lapis_initialize();
    {
      Field initial(std::string("initial Burgers field"));
      const HostField initialHost = initialize(initial);
      const HostField reference = referenceSolution(initialHost);
      const double checksum = validateAndChecksum(initial, reference);
      const auto measurement =
          lapis::benchmark::measure(options, [&] { return burgers(initial); });
      lapis::benchmark::printResult("burgers", LAPIS_BENCHMARK_VARIANT, options,
                                    measurement, checksum);
    }
    lapis_finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
