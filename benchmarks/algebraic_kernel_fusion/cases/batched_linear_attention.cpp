#include LAPIS_BENCHMARK_MODULE

#include "../BenchmarkSupport.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t batches = 8;
constexpr std::size_t queryLength = 256;
constexpr std::size_t keyLength = 1024;
constexpr std::size_t featureSize = 64;
constexpr std::size_t valueSize = 64;

using Queries = LAPIS::DualView<float[batches][queryLength][featureSize],
                                Kokkos::LayoutRight>;
using Keys = LAPIS::DualView<float[batches][keyLength][featureSize],
                             Kokkos::LayoutRight>;
using Values =
    LAPIS::DualView<float[batches][keyLength][valueSize], Kokkos::LayoutRight>;

void initialize(Queries &q, Keys &k, Values &v) {
  auto qHost = q.host_view();
  auto kHost = k.host_view();
  auto vHost = v.host_view();
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t token = 0; token < queryLength; ++token) {
      for (std::size_t feature = 0; feature < featureSize; ++feature) {
        const int value =
            static_cast<int>((batch * 19 + token * 11 + feature * 7 + 3) % 31);
        qHost(batch, token, feature) =
            static_cast<float>(value - 15) * 0.0078125F;
      }
    }
    for (std::size_t token = 0; token < keyLength; ++token) {
      for (std::size_t feature = 0; feature < featureSize; ++feature) {
        const int value =
            static_cast<int>((batch * 23 + token * 5 + feature * 13 + 1) % 29);
        kHost(batch, token, feature) =
            static_cast<float>(value - 14) * 0.0078125F;
      }
      for (std::size_t output = 0; output < valueSize; ++output) {
        const int value =
            static_cast<int>((batch * 7 + token * 17 + output * 3 + 9) % 37);
        vHost(batch, token, output) =
            static_cast<float>(value - 18) * 0.00390625F;
      }
    }
  }
  q.modifyHost();
  k.modifyHost();
  v.modifyHost();
}

double validateAndChecksum(Queries &q, Keys &k, Values &v) {
  auto result = batched_linear_attention(q, k, v);
  Kokkos::fence();
  result.syncHost();
  const auto qHost = q.host_view();
  const auto kHost = k.host_view();
  const auto vHost = v.host_view();
  const auto resultHost = result.host_view();

  std::vector<double> ktv(batches * featureSize * valueSize, 0.0);
  auto ktvAt = [&](std::size_t batch, std::size_t feature,
                   std::size_t output) -> double & {
    return ktv[(batch * featureSize + feature) * valueSize + output];
  };
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t feature = 0; feature < featureSize; ++feature) {
      for (std::size_t output = 0; output < valueSize; ++output) {
        for (std::size_t token = 0; token < keyLength; ++token) {
          ktvAt(batch, feature, output) +=
              static_cast<double>(kHost(batch, token, feature)) *
              vHost(batch, token, output);
        }
      }
    }
  }

  double checksum = 0.0;
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t token = 0; token < queryLength; ++token) {
      for (std::size_t output = 0; output < valueSize; ++output) {
        double expected = 0.0;
        for (std::size_t feature = 0; feature < featureSize; ++feature) {
          expected += static_cast<double>(qHost(batch, token, feature)) *
                      ktvAt(batch, feature, output);
        }
        const double actual = resultHost(batch, token, output);
        const double tolerance = 4.0e-3 + 4.0e-4 * std::abs(expected);
        if (std::abs(actual - expected) > tolerance)
          throw std::runtime_error("batched attention validation failed");
        checksum += actual;
      }
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
          options, [&] { return batched_linear_attention(q, k, v); });
      lapis::benchmark::printResult("batched_linear_attention",
                                    LAPIS_BENCHMARK_VARIANT, options,
                                    measurement, checksum);
    }
    lapis_finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
