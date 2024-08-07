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
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_CCACHE_BUILD=ON -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DCMAKE_INSTALL_PREFIX="$WORKSPACE/llvmInstall" -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=MinSizeRel -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE:FILEPATH=`which python3`

[[ -f llvmBuild/lib/cmake/llvm/LLVMConfig.cmake && \
   -f llvmBuild/lib/cmake/mlir/MLIRConfig.cmake ]] ||
cmake --build llvmBuild --target llvm-headers \
  llvm-libraries  \
  mlir-headers mlir-libraries mlir-opt \
  mlir-translate

[[ -f lapisBuild/Makefile ]] || \
cmake -S LAPIS -B lapisBuild \
  -DLLVM_TARGETS_TO_BUILD="Native" \
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
