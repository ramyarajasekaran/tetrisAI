import sys
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#os.chdir(r'learned/saved_agents')
with open(sys.argv[1], 'r') as file:
    line = file.readline()
    avgs = []
    episode = []
    while line:
        if "%" in line:
            if "100%" not in line:
                tokens = str.split(line)
                avgs.append(float(tokens[-3]))
                episode.append((int(tokens[-6].split('/')[0])))
        line = file.readline()
    print("avg: "  +str(np.mean(avgs)))
    print("initial: " + str(avgs[0]))
    print("final: " + str(avgs[-1]))
    plt.plot(episode,avgs)
    plt.show()
file.close()