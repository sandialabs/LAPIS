#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import csv

matplotlib.use('Agg')

datafile = open('batched_gemm.csv')
datacsv = csv.reader(datafile, delimiter=' ')

header = next(datacsv)
print("CSV header:", header)

n = []
gr = []
h100 = []
mi300 = []

# Just extract and plot the running time (tuples are matrix, nthreads, time)
for row in datacsv:
    print(row)
    n.append(int(row[0]))
    gr.append(float(row[1]))
    h100.append(float(row[2]))
    mi300.append(float(row[3]))

fig, ax = plt.subplots()

# For each matrix, gather the scaling efficiency at each number of threads
plt.plot(n, gr, color='tab:blue', marker='o', linestyle='-', label = "Intel 6980P, 256 cores")
plt.plot(n, h100, color='tab:green', marker='*', linestyle='-', label = "NVIDIA H100")
plt.plot(n, mi300, color='tab:red', marker='^', linestyle='-', label = "AMD MI300A")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Matrix dimension N')
ax.set_title('Batched matrix multiplication: $2^{20}$ pairs of NxN matrices')
ax.set_ylabel('Mean time (s)')
plt.legend()
#ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(range(2, 33, 2))
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax.xaxis.set_tick_params(which='minor', bottom=False)

#fig.tight_layout()
fig = matplotlib.pyplot.gcf()
#fig.set_size_inches(7, 10)
#fig.subplots_adjust(left=0.2)
plt.savefig('batched_gemm.png', dpi=400)

