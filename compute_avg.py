import sys
import os
import numpy as np
from scipy import stats

#os.chdir(r'learned/saved_agents')
with open(sys.argv[1], 'r') as file:
    line = file.readline()
    avgs = []
    while line:
        if "%" in line:
            if "100%" not in line:
                tokens = str.split(line)
                avgs.append(float(tokens[-3]))
        line = file.readline()
    print("avg: "  +str(np.mean(avgs)))
    print("initial: " + str(avgs[0]))
    print("final: " + str(avgs[-1]))
file.close()