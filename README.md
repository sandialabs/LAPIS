# LAPIS

Linear Algebra Performance through Intermediate Subprograms (LAPIS) is a compiler infrastructure based on MLIR for linear algebra that targets both high productivity and performance portability.

## Configuring and Building

### Set up workspace directory, clone repositories
```
# Can have any name, but Workspace is used here
mkdir Workspace
cd Workspace
export WORKSPACE=`pwd`

# Get LLVM with a specific version
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
git checkout b6603e1bf11d
cd ..

# Clone LAPIS
git clone git@github.com:tensor-compilers/LAPIS.git

# Clone Kokkos (master branch contains latest release)
git clone -b master git@github.com:kokkos/kokkos.git

# Create build/install directories
mkdir lapisBuild
mkdir llvmBuild
mkdir llvmInstall
```
### Configure, build and install LLVM/MLIR
```
cd llvmBuild
# Note: python3 must be in PATH. If "python" points to a modern
# python 3.x, then Python3_EXECUTABLE can point to `which python` instead.
cmake \
   -DCMAKE_INSTALL_PREFIX="$WORKSPACE/llvmInstall" \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=MinSizeRel \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE:FILEPATH=`which python3` \
   ../llvm-project/llvm
make -j16 install
cd ..
```
### Configure and build LAPIS
```
# Python3_EXECUTABLE should match the one used to configure LLVM
cmake \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_PREFIX_PATH="$WORKSPACE/llvmInstall" \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=`which python3` \
   ../LAPIS
make -j16
cd ..
```

### Build and install Kokkos
```
mkdir kokkosBuild
mkdir kokkosInstall
cd kokkosBuild
# This example enables only the Serial backend, but any backend can be used
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DCMAKE_INSTALL_PREFIX=$WORKSPACE/kokkosInstall \
  ../kokkos
make -j16 install
cd ..
```

### Finish setting up environment
```
# WORKSPACE is already be set after the above instructions,
# but must be set again for each new terminal session
export WORKSPACE=`pwd`
export LAPIS_SRC=$WORKSPACE/LAPIS

export LLVM_INS=$WORKSPACE/llvmInstall
export KOKKOS_ROOT=$WORKSPACE/kokkosInstall

# Uncomment this line for Linux:
# export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.so
# Uncomment this line for MacOS:
# export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.dylib

export PYTHONPATH=${LLVM_INS}/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=${WORKSPACE}/lapisBuild/python_packages/lapis:$PYTHONPATH
# This is just for LLVM's custom FileCheck (used for running our tests)
export PATH=$PATH:${WORKSPACE}/llvmBuild/bin
```
### Run an example
```
cd $WORKSPACE/LAPIS/examples
python3 spmv_5.py
```
### Run dialect tests
Prerequisite: install ``lit`` testing utility
```
pip install --user lit
```
Run all tests:
```
cd $WORKSPACE/lapisBuild/mlir/test
lit -v .
```
Run tests for a specific dialect (e.g. Kokkos):
```
cd $WORKSPACE/lapisBuild/mlir/test/Dialect
lit -v Kokkos
```

<!--
TODO: for when torch-mlir support is restored

#export TORCH_MLIR_SRC=/home/knliege/dev/llvm-kokkos/torch-mlir-kokkos
#export TORCH_MLIR_INS=/home/knliege/local/torch-mlir
#export PYTHONPATH=${TORCH_MLIR_INS}/python_packages/torch_mlir:$PYTHONPATH
-->

## Developer Guide
### Adding tests
see [AddingNewTests.txt](mlir/test/AddingNewTests.txt).
