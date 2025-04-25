#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np

timesLabeled = {
    'Intel 6980P, 256 cores': 0.0669568,
    'NVIDIA H100':            0.0607858,
    'AMD MI300A':             0.0490257
}

x = 0
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

colors = ['tab:blue', 'tab:green', 'tab:red']

for attribute, measurement in timesLabeled.items():
    offset = 0.2 * multiplier
    ax.bar([x + offset], [measurement], width, label=attribute, color=colors[multiplier])
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)')
ax.set_title('ResNet18 Inference: Batch of 8 Images')
ax.set_xticks([0 + 0.2 * i for i in range(3)], ['Intel CPU', 'H100', 'MI300A'])
ax.legend(loc='upper right', ncols=1, fontsize='large')
#plt.xticks(rotation=45)
#ax.set_ylim(0, 250)

plt.savefig('resnet.png', dpi=400)

