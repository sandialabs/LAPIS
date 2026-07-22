#ifndef LAPIS_BENCHMARKS_ALGEBRAIC_KERNEL_FUSION_BENCHMARK_SUPPORT_HPP
#define LAPIS_BENCHMARKS_ALGEBRAIC_KERNEL_FUSION_BENCHMARK_SUPPORT_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lapis::benchmark {

struct Options {
  std::size_t warmup = 3;
  std::size_t iterations = 20;
};

inline std::size_t parseCount(const char *value, std::string_view option) {
  try {
    std::size_t parsedCharacters = 0;
    const unsigned long parsed = std::stoul(value, &parsedCharacters);
    if (value[parsedCharacters] != '\0' || parsed == 0)
      throw std::invalid_argument("not a positive integer");
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception &) {
    throw std::runtime_error(std::string(option) +
                             " requires a positive integer");
  }
}

inline Options parseOptions(int argc, char **argv) {
  Options options;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string_view option = argv[argument];
    if (option == "--warmup" && argument + 1 < argc) {
      options.warmup = parseCount(argv[++argument], option);
      continue;
    }
    if (option == "--iterations" && argument + 1 < argc) {
      options.iterations = parseCount(argv[++argument], option);
      continue;
    }
    if (option == "--help") {
      std::cout << "Usage: benchmark [--warmup N] [--iterations N]\n";
      std::exit(0);
    }
    throw std::runtime_error("unknown or incomplete option: " +
                             std::string(option));
  }
  return options;
}

struct Measurement {
  double minimumSeconds;
  double medianSeconds;
  double meanSeconds;
};

template <typename Operation>
Measurement measure(const Options &options, Operation &&operation) {
  for (std::size_t iteration = 0; iteration < options.warmup; ++iteration) {
    [[maybe_unused]] auto result = operation();
    Kokkos::fence();
  }

  std::vector<double> samples;
  samples.reserve(options.iterations);
  for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
    Kokkos::Timer timer;
    [[maybe_unused]] auto result = operation();
    Kokkos::fence();
    samples.push_back(timer.seconds());
  }

  std::sort(samples.begin(), samples.end());
  const double median =
      samples.size() % 2 == 0
          ? (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]) /
                2.0
          : samples[samples.size() / 2];
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  return Measurement{samples.front(), median, mean};
}

inline void printResult(std::string_view benchmark, std::string_view variant,
                        const Options &options, const Measurement &measurement,
                        double checksum) {
  std::cout << std::setprecision(12) << "RESULT," << benchmark << ',' << variant
            << ',' << Kokkos::DefaultExecutionSpace::name() << ','
            << options.warmup << ',' << options.iterations << ','
            << measurement.minimumSeconds << ',' << measurement.medianSeconds
            << ',' << measurement.meanSeconds << ',' << checksum << '\n';
}

} // namespace lapis::benchmark

#endif
