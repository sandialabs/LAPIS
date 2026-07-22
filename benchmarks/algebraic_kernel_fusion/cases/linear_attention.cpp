#include LAPIS_BENCHMARK_MODULE

#include "../BenchmarkSupport.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t queryLength = 256;
constexpr std::size_t keyLength = 1024;
constexpr std::size_t featureSize = 64;
constexpr std::size_t valueSize = 64;

using Queries =
    LAPIS::DualView<float[queryLength][featureSize], Kokkos::LayoutRight>;
using Keys =
    LAPIS::DualView<float[keyLength][featureSize], Kokkos::LayoutRight>;
using Values =
    LAPIS::DualView<float[keyLength][valueSize], Kokkos::LayoutRight>;

void initialize(Queries &q, Keys &k, Values &v) {
  auto qHost = q.host_view();
  auto kHost = k.host_view();
  auto vHost = v.host_view();
  for (std::size_t token = 0; token < queryLength; ++token) {
    for (std::size_t feature = 0; feature < featureSize; ++feature) {
      const int value = static_cast<int>((token * 11 + feature * 7 + 3) % 31);
      qHost(token, feature) = static_cast<float>(value - 15) * 0.0078125F;
    }
  }
  for (std::size_t token = 0; token < keyLength; ++token) {
    for (std::size_t feature = 0; feature < featureSize; ++feature) {
      const int value = static_cast<int>((token * 5 + feature * 13 + 1) % 29);
      kHost(token, feature) = static_cast<float>(value - 14) * 0.0078125F;
    }
    for (std::size_t output = 0; output < valueSize; ++output) {
      const int value = static_cast<int>((token * 17 + output * 3 + 9) % 37);
      vHost(token, output) = static_cast<float>(value - 18) * 0.00390625F;
    }
  }
  q.modifyHost();
  k.modifyHost();
  v.modifyHost();
}

double validateAndChecksum(Queries &q, Keys &k, Values &v) {
  auto result = linear_attention(q, k, v);
  Kokkos::fence();
  result.syncHost();

  const auto qHost = q.host_view();
  const auto kHost = k.host_view();
  const auto vHost = v.host_view();
  const auto resultHost = result.host_view();

  std::vector<double> ktv(featureSize * valueSize, 0.0);
  for (std::size_t feature = 0; feature < featureSize; ++feature) {
    for (std::size_t output = 0; output < valueSize; ++output) {
      for (std::size_t token = 0; token < keyLength; ++token) {
        ktv[feature * valueSize + output] +=
            static_cast<double>(kHost(token, feature)) * vHost(token, output);
      }
    }
  }

  double checksum = 0.0;
  for (std::size_t token = 0; token < queryLength; ++token) {
    for (std::size_t output = 0; output < valueSize; ++output) {
      double expected = 0.0;
      for (std::size_t feature = 0; feature < featureSize; ++feature) {
        expected += static_cast<double>(qHost(token, feature)) *
                    ktv[feature * valueSize + output];
      }
      const double actual = resultHost(token, output);
      const double tolerance = 3.0e-3 + 3.0e-4 * std::abs(expected);
      if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("linear-attention validation failed at (" +
                                 std::to_string(token) + ", " +
                                 std::to_string(output) + ")");
      }
      checksum += actual;
    }
  }
  return checksum;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = lapis::benchmark::parseOptions(argc, argv);
    lapis_initialize();
    {
      Queries q(std::string("Q"));
      Keys k(std::string("K"));
      Values v(std::string("V"));
      initialize(q, k, v);

      const double checksum = validateAndChecksum(q, k, v);
      const auto measurement = lapis::benchmark::measure(
          options, [&] { return linear_attention(q, k, v); });
      lapis::benchmark::printResult("linear_attention", LAPIS_BENCHMARK_VARIANT,
                                    options, measurement, checksum);
    }
    lapis_finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
