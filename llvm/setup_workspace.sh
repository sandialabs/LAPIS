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

# Clone Kokkos
[[ -d kokkos/.git ]] || git clone -b master git@github.com:kokkos/kokkos.git

# Create build/install directories
mkdir -p llvmBuild
mkdir -p llvmInstall
mkdir -p lapisBuild
mkdir -p kokkosBuild
mkdir -p kokkosInstall
