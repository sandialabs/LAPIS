#!/usr/bin/env zsh

# Get LLVM with a specific version
if [[ ! -d llvm-project/.git ]]; then
  git clone --depth=1 --branch llvm-upstream \
    git@github.com:tensor-compilers/llvm-project.git
fi

# Clone LAPIS
if [[ ! -d LAPIS/.git ]]; then
  git clone git@github.com:tensor-compilers/LAPIS.git
  pushd LAPIS; git checkout vmiheer/main; popd
fi

if [[ ! -d vmiheer-mlir-playground/.git ]]; then
  git clone git@github.com:tensor-compilers/vmiheer-mlir-playground.git
  pushd vmiheer-mlir-playground; git checkout lapis-main; popd
fi

if [[ ! -d parttensor_mpi_backend/.git ]]; then
  git clone git@github.com:tensor-compilers/parttensor_mpi_backend.git
  pushd parttensor_mpi_backend; git checkout lapis-main; popd
fi

# Clone Kokkos
[[ -d kokkos/.git ]] || git clone --depth 1 -b master \
  git@github.com:kokkos/kokkos.git

# Create build/install directories
mkdir -p llvmBuild
mkdir -p lapisBuild
mkdir -p kokkosBuild
mkdir -p kokkosInstall
