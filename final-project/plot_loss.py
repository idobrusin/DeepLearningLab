import numpy as np
from matplotlib import pyplot as plt

plt.axis([0, 1000, 0, 10])
plt.xlabel("# Steps")
plt.ylabel("Loss")

plt.plot(np.loadtxt("data/logs/run_7-tag-loss.csv", delimiter=",", skiprows=1), label="Loss")

plt.legend()
plt.savefig("data/images/loss.png")
plt.clf()
