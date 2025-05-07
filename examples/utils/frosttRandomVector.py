#!/usr/bin/env python

import sys
import random
import numpy as np
from scipy.io import mmread

if len(sys.argv) != 3:
    print("Usage: ./frosttRandomVector.py <output.tns> <N>")
    quit()

outfile = sys.argv[1]
n = int(sys.argv[2])

os = open(outfile, "w")
os.write("# Extended FROSTT format:\n")
os.write("# rank number-non-zero-elements\n")
os.write("# dimension-sizes\n")
os.write("1 %d\n" % n)
os.write("%d\n" % n)
for i in range(n):
    # Note: scipy coo is 0-based numbered, but FROSTT is 1-based
    os.write("%d %f\n" % (i + 1, random.random()))
os.close()
