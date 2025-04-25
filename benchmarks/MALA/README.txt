be_model.zip contains the complete pre-trained DFT model for beryllium atoms.
We include the exact model version used for the experiments in this paper, but
the most up-to-date model is provided by the MALA developers at github.com/mala-project/test-data.

snap_descriptors.txt contains the raw preprocessed atom descriptors from a
LAMMPS+MALA example simulation. This serves as the model input.

On any machine with LAPIS installed, install the MALA
python package from source (other recent
versions of MALA are likely to work, but this is the version used for this paper).

  git clone https://github.com/mala-project/mala.git
  cd mala
  git checkout 575dc55f81381f305b387d261fb2fc6c14ac38ea
  pip install -e .

Then return to this directory and run:

  python simple_loader.py

to load this model and compile it to C++ (but without executing it).
This produces the C++ file forward_snap.hpp.
