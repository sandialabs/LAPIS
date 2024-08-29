module load cmake/3.26.0.lua
module load gcc/13.1.0
module load ninja/1.11.1.lua
module load ccache/4.6.1.lua
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
export MLIR_Workspace=$(readlink -f ${0:a:h}/..)
