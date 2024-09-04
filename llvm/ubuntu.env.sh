#!/usr/bin/env zsh

export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_ROOT=/usr/local/cuda-12.1
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
source $WORKSPACE/LAPIS/llvm/build_workspace.sh
