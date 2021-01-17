# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data5.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.

#C=c, kernel=kernel,degree=degree, gamma=gamma

model = SVC(C=10000.0,  kernel='rbf', degree= 0, gamma= 10.0) #acc 1

#model = SVC(C=10000.0,  kernel='rbf', degree= 0, gamma= 1.0) #.98
#model = SVC(C=1.0,  kernel='rbf', degree= 0, gamma= 10.0) #acc .91

#model = SVC(C=1000.0,  kernel='rbf', degree= 0, gamma= 1.0) #.875

#model = SVC(C=10.0,  kernel='rbf', degree= 0, gamma= 1.0) #.81
# the original answer -- a bit simpler: model = SVC(kernel='rbf', gamma=27)

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred =  model.predict(X)

# TODO: visualize kato tuk: https://ryanwingate.com/intro-to-machine-learning/supervised/support-vector-machine-implementations/

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)

print(acc)