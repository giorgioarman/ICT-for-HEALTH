import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# close all figures
plt.close('all')

# OPENING THE FILE WITH THE RATIOs
fileOut = 'ratio.txt'
txt = open(fileOut, 'r')
string = txt.read()
strings = string.split('\n')
lowRatios = []
medRatios = []
melRatios = []
min = +10
max = -10
for i in strings:
    try:
        img, ratio = i.split(',')
        ratio = float(ratio)
        if ratio < min:
            min = ratio
        if ratio > max:
            max = ratio
        if img.startswith('low'):
            lowRatios.append(ratio)
        elif img.startswith('med'):
            medRatios.append(ratio)
        else:
            melRatios.append(ratio)
    except:
        pass
lowRatios.sort()
medRatios.sort()
melRatios.sort()

plt.figure(1)
bins = np.arange(min-0.5, max+0.5, 0.25)
plt.hist([lowRatios, medRatios, melRatios], bins, histtype='bar', alpha=0.5, label=['low risk', 'medium risk', 'melanoma'])
plt.xticks(bins)
plt.xlim(min-0.5, max+0.5,)
plt.grid(True)
plt.title("Histogram: Ratio")
plt.xlabel('ratio')
plt.ylabel('frequency')
plt.legend()
plt.savefig('histograph.png')
plt.show()
