import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
# Recall that the perceptron step works as follows. For a point with coordinates (p,q)(p,q), label yy, and prediction given by the equation \hat{y} = step(w_1x_1 + w_2x_2 + b) 
# y^ =step(w1 x1 +w2 x2	 +b):

# If the point is correctly classified, do nothing.
# If the point is classified positive, but it has a negative label, subtract \alpha p, \alpha q,αp,αq, and \alphaα from w_1, w_2, and bb respectively.
# If the point is classified negative, but it has a positive label, add \alpha p, \alpha q,αp,αq, and \alphaα to w_1, w_2, and bb respectively.


    # Fill in code

    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate

    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 50):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

#data = pd.read_csv('data3.csv', header = None)
data = np.loadtxt('data3.csv', delimiter = ',')
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
X = data[:, :-1]
y = data[:, -1]

lines = trainPerceptronAlgorithm(X, y)

# print('Epoch :\t\tW\t\tB')
# for n, line in enumerate(lines):
#     print('{}:\t\t{}\t\t{}'
#           .format(str(n+1).zfill(2),
#                   round(line[0][0],3), 
#                   round(line[1][0],3)))

plt.figure()

X_min = X[:,:1].min()
X_max = X[:,:1].max()

counter = len(lines)
for w, b in lines:
    counter -= 1
    color = [1 - 0.91 ** counter for _ in range(3)]
    plt.plot([X_min-0.5, X_max+0.5],
             [(X_min-0.5) * w + b, (X_max+0.5) * w + b],
             color=color,
             linewidth=0.75)
#plt.show()
    
#plt.figure()

plt.scatter(X[:50,:1], 
            X[:50,1:], 
            c = 'blue',
            zorder=3)
plt.scatter(X[50:,:1], 
            X[50:,1:], 
            c = 'red',
            zorder=3)

plt.gca().set_xlim([-0.5,1.5])
plt.gca().set_ylim([-0.5,1.5])


plt.show()