import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data3.csv', delimiter = ',')
X = data[:, :-1]
y = data[:, -1]


plt.scatter(X, y, c = 'red', zorder=3)