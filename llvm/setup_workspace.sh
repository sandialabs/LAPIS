#!/usr/bin/env zsh

# Get LLVM with a specific version
if [[ ! -d llvm-project/.git ]]; then
  git clone git@github.com:llvm/llvm-project.git
  cd llvm-project
  git checkout 4acc3ffbb0af
  cd ..
fi

# Clone LAPIS
[[ -d LAPIS/.git ]] || \
  git clone git@github.com:tensor-compilers/LAPIS.git

[[ -d vmiheer-mlir-playground/.git ]] || \
  git clone git@github.com:tensor-compilers/vmiheer-mlir-playground.git
[[ -d parttensor_mpi_backend/.git ]] || \
  git clone git@github.com:tensor-compilers/parttensor_mpi_backend.git

# Clone Kokkos
[[ -d kokkos/.git ]] || git clone -b master git@github.com:kokkos/kokkos.git

# Create build/install directories
mkdir -p llvmBuild
mkdir -p llvmInstall
mkdir -p lapisBuild
mkdir -p kokkosBuild
mkdir -p kokkosInstall
