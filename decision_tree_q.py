# Import statements 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Read the data.
data = np.asarray(pd.read_csv('data4.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(random_state=0)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test

# TODO: Fit the model.
#model.fit(X_train, y_train)
model.fit(X, y)

#score = model.score(X_test, y_test)

# TODO: Make predictions. Store them in the variable y_pred.
#y_pred = model.predict(X_test)
y_pred = model.predict(X)



# TODO: Calculate the accuracy and assign it to the variable acc.
#acc = accuracy_score(y_test, y_pred)
acc = accuracy_score(y, y_pred)
print(acc)

# Training the model
model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=10)
model.fit(X_train, y_train)

# # Making predictions
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)

# # Calculating accuracies
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# print('The training accuracy is', train_accuracy)
# print('The test accuracy is', test_accuracy)