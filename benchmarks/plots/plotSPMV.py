#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np

#nnz = [6940000, 50757768, 17588875, 41005206, 37138461, 21005389, 4817870, 60917445, 64531701, 77651847]

# sorted:

# apache2 Laplace3D_100 af_shell7 StocF-1465 PFlow_742 Emilia_923 Elasticity3D_60 Hook_1498 Serena audikw_1 

#n =   [715176,  1000000, 504855,   1465137,  742793,   923136,   648000,   1498023,  1391349,  943695]
#nnz = [4817870, 6940000, 17588875, 21005389, 37138461, 41005206, 50757768, 60917445, 64531701, 77651847]
n =   [1465137,  742793,   923136,   648000,   1498023,  1391349,  943695]
nnz = [21005389, 37138461, 41005206, 50757768, 60917445, 64531701, 77651847]

# Ideal memory traffic per call, in bytes
bytesPerCall = [2 * 8 * n[i] + 12 * nnz[i] + (n[i]+1) * 4 for i in range(len(n))]

# Bandwidth (bytes/sec) for the 3 devices (intel, nv, amd)
bw = [0.219e12, 3.092e12, 3.408e12]

datafile = open('spmv_reduced.csv')
datacsv = csv.reader(datafile, delimiter=' ')

header = next(datacsv)
print("CSV header:", header)

matrices = []
times = [[], [], []]
i = 0
for row in datacsv:
    print(row)
    matrices.append(row[0])
    for j in range(3):
        time = float(row[1+j])
        achieved = bytesPerCall[i] / time
        times[j].append(achieved / bw[j])
    i += 1

#matrices = ("Laplace3D_100", "Elasticity3D_60", "af_shell7")
timesLabeled = {
    'Intel 6980P, 256 cores': times[0],
    'NVIDIA H100':            times[1],
    'AMD MI300A':             times[2]
}

#matNNZ = {}
#for i, m in enumerate(matrices):
#    matNNZ[m] = nnz[i]
#sortedMatNNZ = dict(sorted(matNNZ.items(), key=lambda item: item[1]))

#print("Matrices in increasing order of nnz:")
#print(sortedMatNNZ)

x = np.arange(len(matrices))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

colors = ['tab:blue', 'tab:green', 'tab:red']
hatch = ['/', 'x', 'o']

plt.axhline(y=1, color='k', linestyle='--')

for attribute, measurement in timesLabeled.items():
    offset = width * multiplier
    #rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier], hatch=hatch[multiplier])
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Fraction of Practical Bandwidth')
ax.set_title('SpMV: Fraction of Bandwidth Achieved')
ax.set_xticks(x + width, matrices)
ax.legend(loc='upper left', ncols=3)
plt.xticks(rotation=45)
#ax.set_ylim(0, 250)

plt.savefig('spmv.png', dpi=400)

