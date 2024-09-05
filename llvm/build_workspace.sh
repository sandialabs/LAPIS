function llvm_cmake_linker_options() {
  if [[ $ON_NOTCHPEAK -eq 1 ]]; then
     echo "-DLLVM_ENABLE_LLD=ON"
  else
     echo "-DLLVM_ENABLE_LLD=OFF -DLLVM_USE_LINKER=mold"
  fi
}

[[ -f llvmBuild/build.ninja ]] || \
cmake -GNinja -B llvmBuild -S llvm-project/llvm $(llvm_cmake_linker_options) \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_CCACHE_BUILD=ON -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DCMAKE_INSTALL_PREFIX="$WORKSPACE/llvmInstall" -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=MinSizeRel -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE:FILEPATH=`which python3`

[[ -f llvmBuild/lib/cmake/llvm/LLVMConfig.cmake && \
   -f llvmBuild/lib/cmake/mlir/MLIRConfig.cmake && \
   -e llvmBuild/bin/mlir-opt ]] ||
cmake --build llvmBuild --target llvm-headers \
  llvm-libraries  tools/mlir/python/all \
  mlir-headers mlir-libraries mlir-opt \
  mlir-translate

[[ -f lapisBuild/build.ninja ]] || \
cmake -GNinja -S LAPIS -B lapisBuild $(llvm_cmake_linker_options) \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_CCACHE_BUILD=ON -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$WORKSPACE/llvmBuild" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=`which python3`

[[ -x lapisBuild/bin/lapis-opt ]] || cmake --build lapisBuild -j

function build_kokkos() {
  local ARCH=$1
  KOKKOS_ROOT=$WORKSPACE/kokkos_install_$ARCH
  if [[ ! -d $KOKKOS_ROOT ]]; then
    set -x
    rm -rf $WORKSPACE/kokkosBuild
    mkdir -p $WORKSPACE/kokkosBuild
    cmake -GNinja -DCMAKE_INSTALL_PREFIX=$KOKKOS_ROOT -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_$ARCH=ON -DBUILD_SHARED_LIBS=ON \
      -S $WORKSPACE/kokkos -B $WORKSPACE/kokkosBuild -G Ninja
    cmake --build $WORKSPACE/kokkosBuild -j $NUMPROCS --target install
    set +x
  fi
}

if [[ $ON_NOTCHPEAK -eq 1 ]]; then
  for i in MAXWELL50 MAXWELL52 MAXWELL53 PASCAL60 PASCAL61 VOLTA70 VOLTA72 \
    TURING75 AMPERE80 AMPERE86; do
    build_kokkos $i
  done
else
  build_kokkos TURING75
fi

[[ -f kokkosBuild/build.ninja ]] || \
cmake -GNinja -S kokkos -B kokkosBuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX=$WORKSPACE/kokkosInstall

if [[ ! -x lapisBuild/nv_sm_arch ]]; then
  nvcc $WORKSPACE/kokkos/cmake/compile_tests/cuda_compute_capability.cc \
    -DSM_ONLY -o lapisBuild/nv_sm_arch
fi

export KOKKOS_ROOT=$(readlink -f $(print -C 1 \
  $WORKSPACE/kokkos_install*$(lapisBuild/nv_sm_arch) | head -1))
[[ -d $KOKKOS_ROOT ]] || export KOKKOS_ROOT=$WORKSPACE/kokkos_install
[[ -d $KOKKOS_ROOT ]] || { echo "KOKKOS_ROOT does not exist" }

CMAKE_PREFIX_PATH+=:$WORKSPACE/lapisBuild
export CMAKE_PREFIX_PATH

if [[ ! -f ptMpiBuild/build.ninja ]]; then
  mkdir -p ptMpiBuild
  mkdir -p ptMpiInstall
  cmake -GNinja -S parttensor_mpi_backend -B ptMpiBuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$WORKSPACE/ptMpiInstall
fi

[[ -f ptMpiInstall/share/parttensor_mpi_backend/cmake/\
parttensor_mpi_backendConfig.cmake ]] || \
cmake --build ptMpiBuild --target install

CMAKE_PREFIX_PATH+=:$WORKSPACE/ptMpiInstall
export CMAKE_PREFIX_PATH
export PATH=$WORKSPACE/lapisBuild/bin:$PATH

export LAPIS_SRC=$WORKSPACE/LAPIS

export LLVM_INS=$WORKSPACE/llvmBuild
export PATH=$LLVM_INS/bin:$PATH
# Uncomment this line for Linux:
export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.so
# Uncomment this line for MacOS:
# export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.dylib

export PYTHONPATH=${LLVM_INS}/tools/mlir/python_packages/mlir_core:$PYTHONPATH
# Uncomment this line if using "installed" llvm instead from build directory:
# export PYTHONPATH=${LLVM_INS}/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=${WORKSPACE}/lapisBuild/python_packages/lapis:$PYTHONPATH
