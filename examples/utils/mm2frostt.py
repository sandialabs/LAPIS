#!/usr/bin/env python

import sys
import numpy as np
from scipy.io import mmread

if len(sys.argv) != 3:
    print("Usage: ./mm2frostt.py <input.mtx> <output.tns>")
    quit()

infile = sys.argv[1]
outfile = sys.argv[2]

A = mmread(infile)

# A could be dense if infile's header says "array", but we want COO here
if isinstance(A, np.ndarray):
    A = scipy.sparse.coo_matrix(A)

os = open(outfile, "w")
os.write("# Extended FROSTT format:\n")
os.write("# rank number-non-zero-elements\n")
os.write("# dimension-sizes\n")
os.write("2 %d\n" % A.nnz)
os.write("%d %d\n" % (A.shape[0], A.shape[1]))
for i in range(A.nnz):
    # Note: scipy coo is 0-based numbered, but FROSTT is 1-based
    os.write("%d %d %f\n" % (A.row[i] + 1, A.col[i] + 1, A.data[i]))
os.close()
