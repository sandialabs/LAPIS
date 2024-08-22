#!/usr/bin/env zsh

[[ -f ~/commons.sh ]] && source ~/commons.sh
declare -f spack_setup &> /dev/null && spack_setup

cd ..
spack env activate .
which python | grep ".spack-env/view" 2>&1 > /dev/null || spack install
python -c "import numpy" 2>&1 || pip install -r llvm/notchpeak.requirements.txt

export CC=`which gcc-13`
export CXX=`which g++-13`
export WORKSPACE=$(readlink -f ${0:a:h}/..)
cd $WORKSPACE
source $WORKSPACE/LAPIS/llvm/setup_workspace.sh

[[ -f llvmBuild/build.ninja ]] || \
cmake -GNinja -B llvmBuild -S llvm-project/llvm -DLLVM_ENABLE_LLD=OFF \
  -DLLVM_USE_LINKER=mold \
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
cmake -GNinja -S LAPIS -B lapisBuild -DLLVM_USE_LINKER=mold \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_CCACHE_BUILD=ON -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$WORKSPACE/llvmBuild" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=`which python3`

[[ -x lapisBuild/bin/lapis-opt ]] || cmake --build lapisBuild -j

[[ -f kokkosBuild/build.ninja ]] || \
cmake -GNinja -S kokkos -B kokkosBuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX=$WORKSPACE/kokkosInstall

[[ -f kokkosInstall/lib/cmake/Kokkos/KokkosConfig.cmake ]] || \
cmake --build kokkosBuild --target install
CMAKE_PREFIX_PATH+=:$WORKSPACE/kokkosInstall:$WORKSPACE/llvmBuild
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
export KOKKOS_ROOT=$WORKSPACE/kokkosInstall

export LLVM_INS=$WORKSPACE/llvmBuild
# Uncomment this line for Linux:
export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.so
# Uncomment this line for MacOS:
# export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.dylib

export PYTHONPATH=${LLVM_INS}/tools/mlir/python_packages/mlir_core:$PYTHONPATH
# Uncomment this line if using "installed" llvm instead from build directory:
# export PYTHONPATH=${LLVM_INS}/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=${WORKSPACE}/lapisBuild/python_packages/lapis:$PYTHONPATH
