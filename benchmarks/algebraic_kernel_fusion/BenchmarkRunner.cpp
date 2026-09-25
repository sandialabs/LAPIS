#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

constexpr std::string_view baseline = "baseline";
constexpr std::string_view optimized = "optimized";
constexpr std::string_view managedDirectoryMarker =
    ".lapis-algebraic-benchmark-directory";

const fs::path sourceDirectory = LAPIS_ALGEBRAIC_FUSION_SOURCE_DIR;
const fs::path casesDirectory = sourceDirectory / "cases";
const fs::path repositoryDirectory = LAPIS_REPOSITORY_DIR;

const std::array<std::string_view, 9> capturedEnvironment = {
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "OMP_NUM_THREADS",
    "OMP_PLACES",           "OMP_PROC_BIND",       "ONEAPI_DEVICE_SELECTOR",
    "ROCR_VISIBLE_DEVICES", "SYCL_DEVICE_FILTER",  "ZE_AFFINITY_MASK"};

class MissingPrerequisite : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct BackendInfo {
  std::string cliName;
  std::string kokkosDevice;
  std::string runtimeName;
};

struct Options {
  std::vector<std::string> requestedCases;
  bool listCases = false;
  bool showHelp = false;
  std::size_t warmup = 3;
  std::size_t iterations = 20;
  std::size_t buildJobs = 2;
  fs::path buildDirectory = sourceDirectory / "build";
  std::optional<fs::path> output;
  bool appendOutput = false;
  std::string label;
  std::string notes;
  std::string lapisOpt = "lapis-opt";
  std::string lapisTranslate = "lapis-translate";
  std::string cmake = "cmake";
  std::string cxx;
  fs::path kokkosRoot;
  fs::path supportLibrary;
  std::vector<std::string> cmakeArguments;
  std::string backend;
  bool skipIfUnavailable = false;
};

struct Runtime {
  BackendInfo backend;
  fs::path lapisOpt;
  fs::path lapisTranslate;
  fs::path cmake;
  fs::path cxx;
  fs::path kokkosConfig;
  fs::path supportLibrary;
};

struct Result {
  std::string caseName;
  std::string variant;
  std::string backend;
  std::size_t warmup;
  std::size_t iterations;
  double minimumSeconds;
  double medianSeconds;
  double meanSeconds;
  double checksum;
};

struct RunMetadata {
  std::string timestampUtc;
  std::string label;
  std::string hostname;
  std::string operatingSystem;
  std::string architecture;
  std::string lapisRevision;
  std::string cmakeCxxCompiler;
  std::string cmakeCxxCompilerId;
  std::string cmakeCxxCompilerVersion;
  std::string kokkosVersion;
  std::string kokkosDevices;
  std::string kokkosArch;
  std::string kokkosCxxCompiler;
  std::string kokkosCxxCompilerId;
  std::string kokkosCxxCompilerVersion;
  std::string kokkosConfig;
  std::string runtimeEnvironment;
  std::string notes;
};

std::string getEnvironment(std::string_view name) {
  if (const char *value = std::getenv(std::string(name).c_str()))
    return value;
  return {};
}

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::size_t parsePositiveCount(std::string_view text, std::string_view option) {
  if (text.empty() ||
      !std::all_of(text.begin(), text.end(), [](unsigned char character) {
        return std::isdigit(character);
      }))
    throw std::runtime_error(std::string(option) +
                             " requires a positive integer");
  try {
    const unsigned long long parsed = std::stoull(std::string(text));
    if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max())
      throw std::out_of_range("not a positive size_t");
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception &) {
    throw std::runtime_error(std::string(option) +
                             " requires a positive integer");
  }
}

void printHelp() {
  llvm::outs()
      << "Usage: lapis-algebraic-kernel-fusion-benchmark [options]\n\n"
      << "Required for benchmark runs:\n"
      << "  --backend NAME       expected Kokkos default: cuda, hip, sycl, "
         "openmp, or serial\n"
      << "  --kokkos-root PATH   matching Kokkos install/build/config "
         "directory\n"
      << "  --cxx PATH           matching compiler or compiler wrapper\n\n"
      << "Selection and timing:\n"
      << "  --case NAME          repeatable; use 'all' for every case\n"
      << "  --list-cases         print discovered cases and exit\n"
      << "  --warmup N           warmup calls per variant (default: 3)\n"
      << "  --iterations N       timed calls per variant (default: 20)\n"
      << "  --build-jobs N       parallel build jobs (default: 2)\n\n"
      << "Tools and paths:\n"
      << "  --build-dir PATH\n"
      << "  --lapis-opt PATH\n"
      << "  --lapis-translate PATH\n"
      << "  --cmake PATH\n"
      << "  --support-lib PATH\n"
      << "  --cmake-arg ARG      repeatable extra configure argument\n\n"
      << "Results:\n"
      << "  --output PATH\n"
      << "  --append-output\n"
      << "  --label TEXT\n"
      << "  --notes TEXT\n"
      << "  --skip-if-unavailable\n";
}

std::pair<std::string, std::optional<std::string>>
splitOption(std::string argument) {
  const std::size_t equals = argument.find('=');
  if (equals == std::string::npos)
    return {std::move(argument), std::nullopt};
  return {argument.substr(0, equals), argument.substr(equals + 1)};
}

Options parseOptions(int argc, char **argv) {
  Options options;
  options.cxx = getEnvironment("CXX");
  options.kokkosRoot = getEnvironment("KOKKOS_ROOT");
  options.supportLibrary = getEnvironment("SUPPORT_LIB");
  options.backend = getEnvironment("LAPIS_BENCHMARK_BACKEND");

  auto requireValue = [&](int &index, std::string_view option,
                          std::optional<std::string> inlineValue) {
    if (inlineValue)
      return *inlineValue;
    if (index + 1 >= argc)
      throw std::runtime_error(std::string(option) + " requires a value");
    return std::string(argv[++index]);
  };
  auto rejectValue = [](std::string_view option,
                        const std::optional<std::string> &inlineValue) {
    if (inlineValue)
      throw std::runtime_error(std::string(option) +
                               " does not accept a value");
  };

  for (int index = 1; index < argc; ++index) {
    auto [option, inlineValue] = splitOption(argv[index]);
    if (option == "--help" || option == "-h") {
      rejectValue(option, inlineValue);
      options.showHelp = true;
    } else if (option == "--list-cases") {
      rejectValue(option, inlineValue);
      options.listCases = true;
    } else if (option == "--append-output") {
      rejectValue(option, inlineValue);
      options.appendOutput = true;
    } else if (option == "--skip-if-unavailable") {
      rejectValue(option, inlineValue);
      options.skipIfUnavailable = true;
    } else if (option == "--case") {
      options.requestedCases.push_back(
          requireValue(index, option, std::move(inlineValue)));
    } else if (option == "--warmup") {
      options.warmup = parsePositiveCount(
          requireValue(index, option, std::move(inlineValue)), option);
    } else if (option == "--iterations") {
      options.iterations = parsePositiveCount(
          requireValue(index, option, std::move(inlineValue)), option);
    } else if (option == "--build-jobs") {
      options.buildJobs = parsePositiveCount(
          requireValue(index, option, std::move(inlineValue)), option);
    } else if (option == "--build-dir") {
      options.buildDirectory =
          requireValue(index, option, std::move(inlineValue));
    } else if (option == "--output") {
      options.output = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--label") {
      options.label = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--notes") {
      options.notes = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--lapis-opt") {
      options.lapisOpt = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--lapis-translate") {
      options.lapisTranslate =
          requireValue(index, option, std::move(inlineValue));
    } else if (option == "--cmake") {
      options.cmake = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--cxx") {
      options.cxx = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--kokkos-root") {
      options.kokkosRoot = requireValue(index, option, std::move(inlineValue));
    } else if (option == "--support-lib") {
      options.supportLibrary =
          requireValue(index, option, std::move(inlineValue));
    } else if (option == "--cmake-arg") {
      options.cmakeArguments.push_back(
          requireValue(index, option, std::move(inlineValue)));
    } else if (option == "--backend") {
      options.backend = requireValue(index, option, std::move(inlineValue));
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }

  if (options.appendOutput && !options.output)
    throw std::runtime_error("--append-output requires --output");
  return options;
}

bool isSafeCaseName(std::string_view name) {
  return !name.empty() &&
         std::all_of(name.begin(), name.end(), [](unsigned char character) {
           return std::isalnum(character) || character == '_';
         });
}

std::vector<std::string> discoverCases() {
  std::set<std::string> mlirCases;
  std::set<std::string> driverCases;
  for (const fs::directory_entry &entry :
       fs::directory_iterator(casesDirectory)) {
    if (!entry.is_regular_file())
      continue;
    const std::string extension = entry.path().extension().string();
    if (extension != ".mlir" && extension != ".cpp")
      continue;
    const std::string name = entry.path().stem().string();
    if (!isSafeCaseName(name))
      throw std::runtime_error("unsafe benchmark case name: " + name);
    if (extension == ".mlir")
      mlirCases.insert(name);
    else if (extension == ".cpp")
      driverCases.insert(name);
  }
  if (mlirCases != driverCases)
    throw std::runtime_error(
        "benchmark cases require matching .mlir and .cpp files");
  if (mlirCases.empty())
    throw std::runtime_error("no benchmark cases were found");
  return {mlirCases.begin(), mlirCases.end()};
}

std::vector<std::string>
selectCases(const Options &options, const std::vector<std::string> &available) {
  if (options.requestedCases.empty())
    return {available.front()};
  for (const std::string &caseName : options.requestedCases) {
    if (caseName != "all" && std::find(available.begin(), available.end(),
                                       caseName) == available.end())
      throw std::runtime_error("unknown benchmark case: " + caseName);
  }
  if (std::find(options.requestedCases.begin(), options.requestedCases.end(),
                "all") != options.requestedCases.end())
    return available;

  std::vector<std::string> selected;
  for (const std::string &caseName : options.requestedCases) {
    if (std::find(selected.begin(), selected.end(), caseName) == selected.end())
      selected.push_back(caseName);
  }
  return selected;
}

BackendInfo parseBackend(std::string value) {
  value = toLower(std::move(value));
  if (value == "cuda")
    return {"cuda", "CUDA", "Cuda"};
  if (value == "hip")
    return {"hip", "HIP", "HIP"};
  if (value == "sycl")
    return {"sycl", "SYCL", "SYCL"};
  if (value == "openmp")
    return {"openmp", "OPENMP", "OpenMP"};
  if (value == "serial")
    return {"serial", "SERIAL", "Serial"};
  throw std::runtime_error(
      "backend must be cuda, hip, sycl, openmp, or serial");
}

fs::path resolveExecutable(const std::string &value,
                           std::string_view description) {
  if (value.empty())
    throw MissingPrerequisite(std::string(description) + " was not provided");
  const auto resolved = llvm::sys::findProgramByName(value);
  if (!resolved)
    throw MissingPrerequisite(std::string(description) +
                              " is not executable: " + value);
  if (!llvm::sys::fs::can_execute(*resolved))
    throw MissingPrerequisite(std::string(description) +
                              " is not executable: " + *resolved);
  return fs::absolute(*resolved).lexically_normal();
}

fs::path findKokkosConfig(const fs::path &root) {
  if (root.empty())
    throw MissingPrerequisite(
        "Kokkos root was not provided with --kokkos-root or KOKKOS_ROOT");
  const std::array<fs::path, 5> candidates = {
      root, root / "lib" / "cmake" / "Kokkos",
      root / "lib64" / "cmake" / "Kokkos", root / "share" / "cmake" / "Kokkos",
      root / "cmake_packages" / "Kokkos"};
  for (const fs::path &candidate : candidates) {
    if (fs::is_regular_file(candidate / "KokkosConfig.cmake"))
      return fs::absolute(candidate).lexically_normal();
  }
  throw MissingPrerequisite("KokkosConfig.cmake not found under " +
                            root.string());
}

Runtime resolveRuntime(const Options &options) {
  if (options.backend.empty())
    throw MissingPrerequisite(
        "backend was not provided with --backend or LAPIS_BENCHMARK_BACKEND");
  if (options.supportLibrary.empty())
    throw MissingPrerequisite(
        "support library was not provided with --support-lib or SUPPORT_LIB");
  const fs::path supportLibrary =
      fs::absolute(options.supportLibrary).lexically_normal();
  if (!fs::is_regular_file(supportLibrary))
    throw MissingPrerequisite("support library does not exist: " +
                              supportLibrary.string());
  return Runtime{parseBackend(options.backend),
                 resolveExecutable(options.lapisOpt, "lapis-opt"),
                 resolveExecutable(options.lapisTranslate, "lapis-translate"),
                 resolveExecutable(options.cmake, "CMake"),
                 resolveExecutable(options.cxx, "C++ compiler"),
                 findKokkosConfig(options.kokkosRoot),
                 supportLibrary};
}

std::string renderCommand(const std::vector<std::string> &arguments) {
  std::string rendered;
  for (const std::string &argument : arguments) {
    if (!rendered.empty())
      rendered += ' ';
    if (argument.find_first_of(" \t\"'") == std::string::npos) {
      rendered += argument;
      continue;
    }
    rendered += '"';
    for (char character : argument) {
      if (character == '"' || character == '\\')
        rendered += '\\';
      rendered += character;
    }
    rendered += '"';
  }
  return rendered;
}

int execute(
    const std::vector<std::string> &arguments,
    const std::array<std::optional<llvm::StringRef>, 3> &redirects = {}) {
  llvm::outs() << "+ " << renderCommand(arguments) << '\n';
  llvm::outs().flush();
  llvm::SmallVector<llvm::StringRef, 16> argumentRefs;
  argumentRefs.reserve(arguments.size());
  for (const std::string &argument : arguments)
    argumentRefs.push_back(argument);
  std::string errorMessage;
  const int result =
      llvm::sys::ExecuteAndWait(arguments.front(), argumentRefs, std::nullopt,
                                redirects, 0, 0, &errorMessage);
  if (!errorMessage.empty())
    llvm::errs() << errorMessage << '\n';
  return result;
}

std::string readText(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("could not read " + path.string());
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void writeText(const fs::path &path, std::string_view text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error("could not write " + path.string());
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!output)
    throw std::runtime_error("failed while writing " + path.string());
}

void runChecked(const std::vector<std::string> &arguments) {
  const int result = execute(arguments);
  if (result != 0)
    throw std::runtime_error("command failed (" + renderCommand(arguments) +
                             ")");
}

std::string runCaptured(const std::vector<std::string> &arguments,
                        const fs::path &logsDirectory, std::string_view stage) {
  const fs::path stdoutPath = logsDirectory / (std::string(stage) + ".out");
  const fs::path stderrPath = logsDirectory / (std::string(stage) + ".err");
  const std::string stdoutName = stdoutPath.string();
  const std::string stderrName = stderrPath.string();
  const std::array<std::optional<llvm::StringRef>, 3> redirects = {
      std::nullopt, llvm::StringRef(stdoutName), llvm::StringRef(stderrName)};
  const int result = execute(arguments, redirects);
  const std::string output = readText(stdoutPath);
  const std::string errors = readText(stderrPath);
  if (result != 0)
    throw std::runtime_error("command failed (" + renderCommand(arguments) +
                             "):\n" + output + errors);
  if (!errors.empty())
    llvm::errs() << errors;
  return output;
}

void resetManagedDirectory(const fs::path &path) {
  const fs::path marker = path / managedDirectoryMarker;
  if (fs::exists(path)) {
    if (!fs::is_regular_file(marker))
      throw std::runtime_error("refusing to replace unmanaged directory: " +
                               path.string());
    fs::remove_all(path);
  }
  fs::create_directories(path);
  writeText(marker, "Managed by lapis-algebraic-kernel-fusion-benchmark.\n");
}

bool isWithin(const fs::path &path, const fs::path &directory) {
  const fs::path normalizedPath = fs::absolute(path).lexically_normal();
  const fs::path normalizedDirectory =
      fs::absolute(directory).lexically_normal();
  const auto mismatch =
      std::mismatch(normalizedDirectory.begin(), normalizedDirectory.end(),
                    normalizedPath.begin(), normalizedPath.end());
  return mismatch.first == normalizedDirectory.end();
}

std::string escapeCMake(std::string_view value) {
  std::string escaped;
  for (char character : value) {
    if (character == '\\' || character == '"' || character == '$')
      escaped += '\\';
    escaped += character;
  }
  return escaped;
}

std::string escapeCpp(std::string_view value) {
  std::string escaped;
  for (char character : value) {
    if (character == '\\' || character == '"')
      escaped += '\\';
    escaped += character;
  }
  return escaped;
}

fs::path lowerModule(const Runtime &runtime, const std::string &caseName,
                     std::string_view variant, const fs::path &generated) {
  std::string pipeline = "--sparse-compiler-kokkos=parallelization-strategy="
                         "any-storage-any-loop decompose-sparse-tensors";
  if (variant == optimized)
    pipeline += " algebraic-kernel-fusion";
  const fs::path lowered =
      generated / (caseName + "_" + std::string(variant) + ".mlir");
  runChecked({runtime.lapisOpt.string(), pipeline,
              (casesDirectory / (caseName + ".mlir")).string(), "-o",
              lowered.string()});
  const fs::path module =
      generated / (caseName + "_" + std::string(variant) + "_module.cpp");
  runChecked({runtime.lapisTranslate.string(), lowered.string(), "-o",
              module.string(), "--finalize"});
  return module;
}

std::map<std::string, std::map<std::string, fs::path>>
lowerCases(const Runtime &runtime, const std::vector<std::string> &cases,
           const fs::path &generated) {
  std::map<std::string, std::map<std::string, fs::path>> modules;
  for (const std::string &caseName : cases) {
    modules[caseName][std::string(baseline)] =
        lowerModule(runtime, caseName, baseline, generated);
    modules[caseName][std::string(optimized)] =
        lowerModule(runtime, caseName, optimized, generated);
  }
  return modules;
}

void writeBuildProject(
    const Runtime &runtime, const fs::path &projectDirectory,
    const std::map<std::string, std::map<std::string, fs::path>> &modules) {
  std::ostringstream targets;
  for (const auto &[caseName, variants] : modules) {
    for (const auto &[variant, module] : variants) {
      const std::string target = caseName + "_" + variant;
      const fs::path translationUnit = projectDirectory / (target + ".cpp");
      writeText(translationUnit,
                "#define LAPIS_BENCHMARK_MODULE \"" +
                    escapeCpp(module.string()) + "\"\n" +
                    "#define LAPIS_BENCHMARK_VARIANT \"" + variant +
                    "\"\n#include \"" +
                    escapeCpp((casesDirectory / (caseName + ".cpp")).string()) +
                    "\"\n");
      targets << "add_executable(" << target << " \""
              << escapeCMake(translationUnit.string()) << "\")\n"
              << "target_link_libraries(" << target
              << " PRIVATE Kokkos::kokkos \""
              << escapeCMake(runtime.supportLibrary.string()) << "\")\n"
              << "file(GENERATE OUTPUT \"${CMAKE_BINARY_DIR}/lapis-" << target
              << "-$<CONFIG>.path\" CONTENT \"$<TARGET_FILE:" << target
              << ">\")\n";
    }
  }

  const std::string cmake =
      "cmake_minimum_required(VERSION 3.20 FATAL_ERROR)\n"
      "project(LAPISAlgebraicFusionBenchmark LANGUAGES CXX)\n"
      "set(CMAKE_CXX_STANDARD 17)\n"
      "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
      "set(CMAKE_CXX_EXTENSIONS OFF)\n"
      "set(CMAKE_BUILD_RPATH \"" +
      escapeCMake(runtime.supportLibrary.parent_path().string()) +
      "\")\nfind_package(Kokkos REQUIRED CONFIG PATHS \"" +
      escapeCMake(runtime.kokkosConfig.string()) +
      "\" NO_DEFAULT_PATH)\nlist(FIND Kokkos_DEVICES \"" +
      runtime.backend.kokkosDevice +
      "\" LAPIS_BACKEND_INDEX)\nif(LAPIS_BACKEND_INDEX EQUAL -1)\n"
      "  message(FATAL_ERROR \"Selected backend is not enabled in this Kokkos "
      "package: ${Kokkos_DEVICES}\")\nendif()\n"
      "file(WRITE \"${CMAKE_BINARY_DIR}/lapis-benchmark-build-metadata.txt\"\n"
      "  \"cmake_cxx_compiler=${CMAKE_CXX_COMPILER}\\n\"\n"
      "  \"cmake_cxx_compiler_id=${CMAKE_CXX_COMPILER_ID}\\n\"\n"
      "  \"cmake_cxx_compiler_version=${CMAKE_CXX_COMPILER_VERSION}\\n\"\n"
      "  \"kokkos_version=${Kokkos_VERSION}\\n\"\n"
      "  \"kokkos_devices=${Kokkos_DEVICES}\\n\"\n"
      "  \"kokkos_arch=${Kokkos_ARCH}\\n\"\n"
      "  \"kokkos_cxx_compiler=${Kokkos_CXX_COMPILER}\\n\"\n"
      "  \"kokkos_cxx_compiler_id=${Kokkos_CXX_COMPILER_ID}\\n\"\n"
      "  \"kokkos_cxx_compiler_version=${Kokkos_CXX_COMPILER_VERSION}\\n\")\n" +
      targets.str();
  writeText(projectDirectory / "CMakeLists.txt", cmake);
}

std::map<std::string, std::string>
readBuildMetadata(const fs::path &binaryDirectory) {
  std::map<std::string, std::string> metadata;
  std::istringstream input(
      readText(binaryDirectory / "lapis-benchmark-build-metadata.txt"));
  for (std::string line; std::getline(input, line);) {
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos)
      throw std::runtime_error("malformed build metadata: " + line);
    metadata[line.substr(0, equals)] = line.substr(equals + 1);
  }
  return metadata;
}

std::pair<fs::path, std::map<std::string, std::string>>
prepareBenchmarks(const Options &options, const Runtime &runtime,
                  const std::vector<std::string> &cases) {
  const fs::path buildDirectory = fs::absolute(options.buildDirectory);
  fs::create_directories(buildDirectory);
  const fs::path projectDirectory = buildDirectory / "generated-project";
  const fs::path binaryDirectory = buildDirectory / "cmake-build";
  const fs::path logsDirectory = buildDirectory / "logs";
  if (options.output && (isWithin(*options.output, projectDirectory) ||
                         isWithin(*options.output, binaryDirectory) ||
                         isWithin(*options.output, logsDirectory)))
    throw std::runtime_error(
        "result CSV must be outside runner-managed build subdirectories");
  resetManagedDirectory(projectDirectory);
  resetManagedDirectory(binaryDirectory);
  resetManagedDirectory(logsDirectory);
  const fs::path generated = projectDirectory / "generated";
  fs::create_directory(generated);

  const auto modules = lowerCases(runtime, cases, generated);
  writeBuildProject(runtime, projectDirectory, modules);

  std::vector<std::string> configure = {runtime.cmake.string(),
                                        "-E",
                                        "env",
                                        "--unset=KOKKOS_ROOT",
                                        "--unset=Kokkos_ROOT",
                                        runtime.cmake.string(),
                                        "-S",
                                        projectDirectory.string(),
                                        "-B",
                                        binaryDirectory.string(),
                                        "-DCMAKE_BUILD_TYPE=Release",
                                        "-DCMAKE_CXX_COMPILER=" +
                                            runtime.cxx.string()};
  configure.insert(configure.end(), options.cmakeArguments.begin(),
                   options.cmakeArguments.end());
  runChecked(configure);
  runChecked({runtime.cmake.string(), "--build", binaryDirectory.string(),
              "--config", "Release", "--parallel",
              std::to_string(options.buildJobs)});
  return {binaryDirectory, readBuildMetadata(binaryDirectory)};
}

std::vector<std::string> split(std::string_view text, char delimiter) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = text.find(delimiter, begin);
    fields.emplace_back(text.substr(begin, end - begin));
    if (end == std::string_view::npos)
      return fields;
    begin = end + 1;
  }
}

double parseFiniteDouble(std::string_view text, std::string_view field,
                         bool positive) {
  try {
    std::size_t parsedCharacters = 0;
    const double parsed = std::stod(std::string(text), &parsedCharacters);
    if (parsedCharacters != text.size() || !std::isfinite(parsed) ||
        (positive && parsed <= 0.0))
      throw std::invalid_argument("invalid value");
    return parsed;
  } catch (const std::exception &) {
    throw std::runtime_error("invalid " + std::string(field) + ": " +
                             std::string(text));
  }
}

Result parseResult(std::string_view output) {
  std::optional<std::string> resultLine;
  std::istringstream lines{std::string(output)};
  for (std::string line; std::getline(lines, line);) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.rfind("RESULT,", 0) == 0) {
      if (resultLine)
        throw std::runtime_error("benchmark emitted multiple RESULT lines");
      resultLine = std::move(line);
    }
  }
  if (!resultLine)
    throw std::runtime_error("benchmark did not emit a RESULT line");
  const std::vector<std::string> fields = split(*resultLine, ',');
  if (fields.size() != 10)
    throw std::runtime_error("benchmark RESULT line has the wrong field count");
  return Result{fields[1],
                fields[2],
                fields[3],
                parsePositiveCount(fields[4], "RESULT warmup"),
                parsePositiveCount(fields[5], "RESULT iterations"),
                parseFiniteDouble(fields[6], "minimum time", true),
                parseFiniteDouble(fields[7], "median time", true),
                parseFiniteDouble(fields[8], "mean time", true),
                parseFiniteDouble(fields[9], "checksum", false)};
}

fs::path findBenchmarkExecutable(const fs::path &binaryDirectory,
                                 const std::string &target) {
  const std::string manifestPrefix = "lapis-" + target + "-";
  for (const fs::directory_entry &entry :
       fs::directory_iterator(binaryDirectory)) {
    const std::string fileName = entry.path().filename().string();
    if (!entry.is_regular_file() || fileName.rfind(manifestPrefix, 0) != 0 ||
        entry.path().extension() != ".path")
      continue;
    const fs::path candidate = readText(entry.path());
    if (fs::is_regular_file(candidate) &&
        llvm::sys::fs::can_execute(candidate.string()))
      return candidate;
  }

  const std::array<fs::path, 4> candidates = {
      binaryDirectory / target, binaryDirectory / (target + ".exe"),
      binaryDirectory / "Release" / target,
      binaryDirectory / "Release" / (target + ".exe")};
  for (const fs::path &candidate : candidates) {
    if (fs::is_regular_file(candidate) &&
        llvm::sys::fs::can_execute(candidate.string()))
      return candidate;
  }
  throw std::runtime_error("built benchmark executable not found: " + target);
}

bool equalCaseInsensitive(std::string lhs, std::string rhs) {
  return toLower(std::move(lhs)) == toLower(std::move(rhs));
}

std::array<Result, 2> runCase(const Options &options, const Runtime &runtime,
                              const std::string &caseName,
                              const fs::path &binaryDirectory,
                              const fs::path &logsDirectory) {
  std::array<Result, 2> results;
  const std::array<std::string_view, 2> variants = {baseline, optimized};
  for (std::size_t index = 0; index < variants.size(); ++index) {
    const std::string variant(variants[index]);
    const fs::path executable =
        findBenchmarkExecutable(binaryDirectory, caseName + "_" + variant);
    std::string libraryPath = runtime.supportLibrary.parent_path().string();
#if defined(_WIN32)
    constexpr std::string_view libraryPathVariable = "PATH";
    constexpr char pathSeparator = ';';
#elif defined(__APPLE__)
    constexpr std::string_view libraryPathVariable = "DYLD_LIBRARY_PATH";
    constexpr char pathSeparator = ':';
#else
    constexpr std::string_view libraryPathVariable = "LD_LIBRARY_PATH";
    constexpr char pathSeparator = ':';
#endif
    const std::string existingLibraryPath = getEnvironment(libraryPathVariable);
    if (!existingLibraryPath.empty()) {
      libraryPath += pathSeparator;
      libraryPath += existingLibraryPath;
    }
    const std::string output = runCaptured(
        {runtime.cmake.string(), "-E", "env",
         std::string(libraryPathVariable) + "=" + libraryPath,
         executable.string(), "--warmup", std::to_string(options.warmup),
         "--iterations", std::to_string(options.iterations)},
        logsDirectory, caseName + "_" + variant);
    llvm::outs() << output;
    results[index] = parseResult(output);
    const Result &result = results[index];
    if (result.caseName != caseName || result.variant != variant)
      throw std::runtime_error("benchmark reported the wrong case or variant");
    if (result.warmup != options.warmup ||
        result.iterations != options.iterations)
      throw std::runtime_error("benchmark reported inconsistent run counts");
    if (!equalCaseInsensitive(result.backend, runtime.backend.runtimeName))
      throw std::runtime_error("selected backend " + runtime.backend.cliName +
                               " ran as " + result.backend);
  }

  const double checksumScale = std::max(
      {1.0, std::abs(results[0].checksum), std::abs(results[1].checksum)});
  if (std::abs(results[0].checksum - results[1].checksum) >
      1.0e-4 * checksumScale)
    throw std::runtime_error(caseName + " checksum mismatch between variants");
  const double speedup = results[0].medianSeconds / results[1].medianSeconds;
  std::ostringstream speedupText;
  speedupText << std::fixed << std::setprecision(6) << speedup;
  llvm::outs() << "SUMMARY," << caseName << ",median_speedup,"
               << speedupText.str() << '\n';
  return results;
}

std::string trim(std::string value) {
  const auto notSpace = [](unsigned char character) {
    return !std::isspace(character);
  };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), notSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(),
              value.end());
  return value;
}

std::string getHostname() {
#if defined(_WIN32)
  const std::string hostname = getEnvironment("COMPUTERNAME");
  return hostname.empty() ? "unknown" : hostname;
#else
  std::array<char, 256> hostname{};
  return gethostname(hostname.data(), hostname.size()) == 0 ? hostname.data()
                                                            : "unknown";
#endif
}

std::pair<std::string, std::string> getPlatform() {
#if defined(_WIN32)
  return {"Windows", getEnvironment("PROCESSOR_ARCHITECTURE")};
#else
  struct utsname information{};
  if (uname(&information) != 0)
    return {"unknown", "unknown"};
  return {std::string(information.sysname) + "-" + information.release,
          information.machine};
#endif
}

std::string makeTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          now.time_since_epoch()) %
      std::chrono::seconds(1);
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(6)
         << std::setfill('0') << microseconds.count() << "+00:00";
  return output.str();
}

std::string jsonEscape(std::string_view value) {
  std::ostringstream output;
  for (unsigned char character : value) {
    switch (character) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\b':
      output << "\\b";
      break;
    case '\f':
      output << "\\f";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20)
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned>(character) << std::dec;
      else
        output << static_cast<char>(character);
    }
  }
  return output.str();
}

std::string captureEnvironmentJson() {
  std::map<std::string, std::string> values;
  for (std::string_view name : capturedEnvironment) {
    const std::string value = getEnvironment(name);
    if (!value.empty())
      values[std::string(name)] = value;
  }
  std::ostringstream output;
  output << '{';
  bool first = true;
  for (const auto &[name, value] : values) {
    if (!first)
      output << ", ";
    first = false;
    output << '"' << jsonEscape(name) << "\": \"" << jsonEscape(value) << '"';
  }
  output << '}';
  return output.str();
}

std::string getLapisRevision(const fs::path &logsDirectory) {
  const auto git = llvm::sys::findProgramByName("git");
  if (!git)
    return "unknown";
  try {
    return trim(runCaptured({*git, "-C", repositoryDirectory.string(),
                             "describe", "--always", "--dirty"},
                            logsDirectory, "git-revision"));
  } catch (const std::exception &) {
    return "unknown";
  }
}

std::string metadataValue(const std::map<std::string, std::string> &metadata,
                          std::string_view key) {
  const auto found = metadata.find(std::string(key));
  return found == metadata.end() || found->second.empty() ? "unknown"
                                                          : found->second;
}

RunMetadata
makeRunMetadata(const Options &options, const Runtime &runtime,
                const std::map<std::string, std::string> &buildMetadata,
                const fs::path &logsDirectory) {
  auto [operatingSystem, architecture] = getPlatform();
  return RunMetadata{
      makeTimestamp(),
      options.label.empty() ? getHostname() : options.label,
      getHostname(),
      operatingSystem,
      architecture.empty() ? "unknown" : architecture,
      getLapisRevision(logsDirectory),
      metadataValue(buildMetadata, "cmake_cxx_compiler"),
      metadataValue(buildMetadata, "cmake_cxx_compiler_id"),
      metadataValue(buildMetadata, "cmake_cxx_compiler_version"),
      metadataValue(buildMetadata, "kokkos_version"),
      metadataValue(buildMetadata, "kokkos_devices"),
      metadataValue(buildMetadata, "kokkos_arch"),
      metadataValue(buildMetadata, "kokkos_cxx_compiler"),
      metadataValue(buildMetadata, "kokkos_cxx_compiler_id"),
      metadataValue(buildMetadata, "kokkos_cxx_compiler_version"),
      runtime.kokkosConfig.string(),
      captureEnvironmentJson(),
      options.notes};
}

void printMetadata(const RunMetadata &metadata) {
  llvm::outs() << "Benchmark configuration:\n"
               << "  label: " << metadata.label << '\n'
               << "  LAPIS revision: " << metadata.lapisRevision << '\n'
               << "  C++ compiler: " << metadata.cmakeCxxCompilerId << ' '
               << metadata.cmakeCxxCompilerVersion << " ("
               << metadata.cmakeCxxCompiler << ")\n"
               << "  Kokkos: " << metadata.kokkosVersion
               << "; devices=" << metadata.kokkosDevices
               << "; arch=" << metadata.kokkosArch << '\n'
               << "  Kokkos package: " << metadata.kokkosConfig << '\n'
               << "  runtime environment: " << metadata.runtimeEnvironment
               << '\n';
}

std::string csvEscape(std::string_view value) {
  if (value.find_first_of(",\"\r\n") == std::string_view::npos)
    return std::string(value);
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '"')
      escaped += '"';
    escaped += character;
  }
  escaped += '"';
  return escaped;
}

const std::string csvHeader =
    "timestamp_utc,label,hostname,operating_system,architecture,lapis_revision,"
    "cmake_cxx_compiler,cmake_cxx_compiler_id,cmake_cxx_compiler_version,"
    "kokkos_version,kokkos_devices,kokkos_arch,kokkos_cxx_compiler,"
    "kokkos_cxx_compiler_id,kokkos_cxx_compiler_version,kokkos_config,"
    "runtime_environment,notes,case,variant,backend,warmup,iterations,"
    "minimum_seconds,median_seconds,mean_seconds,checksum";

std::vector<std::string> metadataFields(const RunMetadata &metadata) {
  return {metadata.timestampUtc,
          metadata.label,
          metadata.hostname,
          metadata.operatingSystem,
          metadata.architecture,
          metadata.lapisRevision,
          metadata.cmakeCxxCompiler,
          metadata.cmakeCxxCompilerId,
          metadata.cmakeCxxCompilerVersion,
          metadata.kokkosVersion,
          metadata.kokkosDevices,
          metadata.kokkosArch,
          metadata.kokkosCxxCompiler,
          metadata.kokkosCxxCompilerId,
          metadata.kokkosCxxCompilerVersion,
          metadata.kokkosConfig,
          metadata.runtimeEnvironment,
          metadata.notes};
}

std::string resultRow(const RunMetadata &metadata, const Result &result) {
  std::vector<std::string> fields = metadataFields(metadata);
  fields.insert(fields.end(), {result.caseName, result.variant, result.backend,
                               std::to_string(result.warmup),
                               std::to_string(result.iterations)});
  auto formatDouble = [](double value) {
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
  };
  fields.insert(fields.end(), {formatDouble(result.minimumSeconds),
                               formatDouble(result.medianSeconds),
                               formatDouble(result.meanSeconds),
                               formatDouble(result.checksum)});
  std::ostringstream row;
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0)
      row << ',';
    row << csvEscape(fields[index]);
  }
  return row.str();
}

void validateOutputSchema(const Options &options) {
  if (!options.output || !options.appendOutput ||
      !fs::is_regular_file(*options.output) ||
      fs::file_size(*options.output) == 0)
    return;
  std::ifstream input(*options.output);
  std::string header;
  std::getline(input, header);
  if (!header.empty() && header.back() == '\r')
    header.pop_back();
  if (header != csvHeader)
    throw std::runtime_error("cannot append: existing CSV schema differs");
}

void writeResults(const Options &options, bool append,
                  const RunMetadata &metadata,
                  const std::vector<Result> &results) {
  if (!options.output)
    return;
  const fs::path outputPath = fs::absolute(*options.output);
  if (!outputPath.parent_path().empty())
    fs::create_directories(outputPath.parent_path());
  append = append && fs::is_regular_file(outputPath) &&
           fs::file_size(outputPath) != 0;
  std::ofstream output(
      outputPath, std::ios::out | (append ? std::ios::app : std::ios::trunc));
  if (!output)
    throw std::runtime_error("could not write " + outputPath.string());
  if (!append)
    output << csvHeader << '\n';
  for (const Result &result : results)
    output << resultRow(metadata, result) << '\n';
  if (!output)
    throw std::runtime_error("failed while writing " + outputPath.string());
  llvm::outs() << "WROTE," << outputPath.string() << '\n';
}

int run(const Options &options) {
  const std::vector<std::string> available = discoverCases();
  if (options.listCases) {
    for (const std::string &caseName : available)
      llvm::outs() << caseName << '\n';
    return 0;
  }
  const std::vector<std::string> cases = selectCases(options, available);
  validateOutputSchema(options);
  const Runtime runtime = resolveRuntime(options);
  auto [binaryDirectory, buildMetadata] =
      prepareBenchmarks(options, runtime, cases);
  const fs::path logsDirectory = fs::absolute(options.buildDirectory) / "logs";
  const RunMetadata metadata =
      makeRunMetadata(options, runtime, buildMetadata, logsDirectory);
  printMetadata(metadata);

  bool wroteResults = false;
  for (const std::string &caseName : cases) {
    const auto paired =
        runCase(options, runtime, caseName, binaryDirectory, logsDirectory);
    const std::vector<Result> pairedResults(paired.begin(), paired.end());
    writeResults(options, options.appendOutput || wroteResults, metadata,
                 pairedResults);
    wroteResults = wroteResults || options.output.has_value();
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  try {
    options = parseOptions(argc, argv);
    if (options.showHelp) {
      printHelp();
      return 0;
    }
    return run(options);
  } catch (const MissingPrerequisite &error) {
    llvm::errs() << "SKIP: " << error.what() << '\n';
    return options.skipIfUnavailable ? 77 : 1;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return 1;
  }
}
