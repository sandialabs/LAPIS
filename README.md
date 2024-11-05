# LAPIS

Linear Algebra Performance through Intermediate Subprograms (LAPIS) is a compiler infrastructure based on MLIR for linear algebra that targets both high productivity and performance portability.

## Configuring and Building

### Set up workspace directory
```
# Can have any name, but Workspace is used here
mkdir Workspace
cd Workspace
export WORKSPACE=`pwd`
```

### Get source code and dependencies

* LAPIS: this repository, main branch
* [llvm-project](https://github.com/llvm/llvm-project.git): required, version b6603e1b. 
* [kokkos](https://github.com/kokkos/kokkos): required for building and running the generated C++ code. Any recent release or develop branch will work.
* [torch-mlir](https://github.com/llvm/torch-mlir.git): optional, version 6934ab81. 
  * This includes the correct llvm-project version as a submodule, in ``externals/llvm-project``.
* [mpact](https://github.com/MPACT-ORG/mpact-compiler): optional, version 556009cd. Requires torch-mlir. 
  * This includes the correct torch-mlir version as a submodule, in ``externals/torch-mlir``.
  * This torch-mlir also has the correct llvm-project as a submodule.
* Python: optional for dialect development, lowering and C++ emitter. 3.10+ required for running end-to-end examples from PyTorch.

The remaining instructions assume that environment variables ``LAPIS_SRC``, ``LLVM_SRC``, ``TORCH_MLIR_SRC``, and ``MPACT_SRC``
are set to the paths of these repositories.

The following commands will clone the correct versions of all the repositories and set these environment variables.
```
git clone git@github.com:MPACT-ORG/mpact-compiler
cd mpact-compiler
git checkout 556009cd
git submodule update --init --recursive
export MPACT_SRC=`pwd`
export TORCH_MLIR_SRC="$MPACT_SRC/externals/torch-mlir"
export LLVM_SRC="$TORCH_MLIR_SRC/externals/llvm-project"
git clone git@github.com:sandialabs/LAPIS
cd LAPIS
export LAPIS_SRC=`pwd`
cd ..
git clone -b master git@github.com:kokkos/kokkos
```

Building with ninja is not required but useful as it automatically uses all cores for parallel compilation. Pass ``-Gninja`` to
cmake and then run ``ninja`` instead of ``make``.

### Configure and build
#### Recipe A: build LAPIS against an installation of LLVM/MLIR
This recipe can be used if torch-mlir and mpact are not required.
Since LAPIS is configured separately, you can re-run cmake for LAPIS without triggering an entire
rebuild of LLVM. This is why it's recommended for LAPIS development work on dialects, passes, and the Kokkos emitter.
```
cd $WORKSPACE
# Build and install MLIR
mkdir llvmBuild
mkdir llvmInstall
cd llvmBuild
# Note: here, python3 must be in PATH. If "python" points to a modern
# python 3.x, then Python3_EXECUTABLE can point to `which python` instead.
# Python bindings are optional and can be turned off.
cmake \
   -DCMAKE_INSTALL_PREFIX="$WORKSPACE/llvmInstall" \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=MinSizeRel \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE:FILEPATH=`which python3` \
   $LLVM_SRC/llvm
make install

cd ..
mkdir lapisBuild
cd lapisBuild
# Python3_EXECUTABLE should match the one used to configure LLVM
cmake \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_PREFIX_PATH="$WORKSPACE/llvmInstall" \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=`which python3` \
   -DLAPIS_ENABLE_PART_TENSOR=OFF \
   $LAPIS_SRC
make
cd ..
```
#### Recipe B: build LAPIS in-tree with LLVM/MLIR, and optionally torch-mlir/mpact
This recipe builds LAPIS as an external project with LLVM.
torch-mlir and mpact require this recipe, but torch-mlir and mpact are still optional.
mpact requires torch-mlir, however.
```
# If enabling torch-mlir, need to install Python dependencies first.
# This can be done inside a python virtual env.

cd $TORCH_MLIR_SRC
pip install -r requirements.txt
pip install -r torchvision-requirements.txt

cd $WORKSPACE
mkdir build
cd build

# Base configuration: just LAPIS and LLVM/MLIR
cmake \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="lapis" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_EXTERNAL_LAPIS_SOURCE_DIR="$LAPIS_SRC" \
  -DLAPIS_ENABLE_PART_TENSOR=OFF \
  $LLVM_SRC/llvm

# To also enable torch-mlir, set:
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;lapis" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$TORCH_MLIR_SRC" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \

# To also enable mpact, set:
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;mpact;lapis" \
  -DLLVM_EXTERNAL_MPACT_SOURCE_DIR="$MPACT_SRC" \

# Then run make.
```

### For both recipes: build and install Kokkos
```
cd $WORKSPACE
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
make install
cd ..
# Set KOKKOS_ROOT to the install directory
export KOKKOS_ROOT=$WORKSPACE/kokkosInstall
```

### Finish setting up environment
#### Recipe A
```
# WORKSPACE is already be set after the above instructions,
# but must be set again for each new terminal session
export WORKSPACE=`pwd`

# Set path to the Kokkos installation created above
export KOKKOS_ROOT=$WORKSPACE/kokkosInstall

export LLVM_INS=$WORKSPACE/llvmInstall
# Uncomment this line for Linux:
# export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.so
# Uncomment this line for MacOS:
# export SUPPORTLIB=${LLVM_INS}/lib/libmlir_c_runner_utils.dylib

# Put lapis-opt, lapis-translate in PATH
export PATH=$PATH:$WORKSPACE/lapisBuild/bin

# Only if python bindings were enabled:
export PYTHONPATH=${LLVM_INS}/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=${WORKSPACE}/lapisBuild/python_packages/lapis:$PYTHONPATH
```

#### Recipe B
```
# WORKSPACE is already be set after the above instructions,
# but must be set again for each new terminal session
export WORKSPACE=`pwd`

export KOKKOS_ROOT=$WORKSPACE/kokkosInstall

# Uncomment this line for Linux:
# export SUPPORTLIB=$WORKSPACE/build/lib/libmlir_c_runner_utils.so
# Uncomment this line for MacOS:
# export SUPPORTLIB=$WORKSPACE/build/lib/libmlir_c_runner_utils.dylib

export PATH=$PATH:$WORKSPACE/build/bin

# MLIR, LAPIS, torch-mlir and MPACT each build their own python packages
export PYTHONPATH=$PYTHONPATH:$WORKSPACE/build/tools/mlir/python_packages/mlir_core
export PYTHONPATH=$PYTHONPATH:$WORKSPACE/build/tools/lapis/python_packages/lapis
export PYTHONPATH=$PYTHONPATH:$WORKSPACE/build/tools/torch-mlir/python_packages/torch_mlir
export PYTHONPATH=$PYTHONPATH:$WORKSPACE/build/tools/mpact/python_packages/mpact
```

### Run Kokkos dialect tests
Prerequisite: install ``lit`` testing utility
```
pip install --user lit
```
#### Run tests: recipe A
```
cd $WORKSPACE/lapisBuild/mlir/test
# Just Kokkos dialect tests
lit -v Dialect/Kokkos
```
#### Run tests: recipe B
```
cd $WORKSPACE/build/tools/lapis/mlir/test
lit -v Dialect/Kokkos
```

## Developer Guide
### Adding tests
see [AddingNewTests.txt](mlir/test/AddingNewTests.txt).
