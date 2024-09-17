#!/uufs/chpc.utah.edu/common/home/u1290058/.local/bin/zsh
#
module () {
  eval $($LMOD_CMD zsh "$@") && eval $(${LMOD_SETTARG_CMD:-:} -s sh)
}

export ON_NOTCHPEAK=1
module load cmake/3.26.0.lua
module load ninja/1.11.1.lua
module load ccache/4.6.1.lua
# reloading gcc causes error about loading zlib module
which gcc | grep gcc-11.2.0 &> /dev/null || module load gcc/11.2.0
which mold &> /dev/null || module load mold/2.1.0
. ~/scratchVast/setup/spack/share/spack/setup-env.sh
cd ..
spack env activate .
which python | grep ".spack-env/view" 2>&1 > /dev/null || spack install
python -c "import numpy" &> /dev/null || pip install -r llvm/notchpeak.requirements.txt

export CC=`which gcc`
export CXX=`which g++`
export CCACHE_DIR=`readlink -f ~/scratchVast/mlirWorkspace/ccache`
export PATH=`git rev-parse --show-toplevel`/llvm/build/bin:$PATH
export WORKSPACE=$(readlink -f ${0:a:h}/..)

cd $WORKSPACE
source $WORKSPACE/LAPIS/llvm/setup_workspace.sh
source $WORKSPACE/LAPIS/llvm/build_workspace.sh
