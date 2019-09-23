import numpy as np
import decision_trees as dt

X = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
Y = np.array([[1], [1], [0]])
max_depth = 2

DT = dt.DT_train_binary(X, Y, max_depth)
test_acc = dt.DT_test_binary(X, Y, DT)
print(test_acc)


training_features = np.array([[0,1], [0,0], [1,0], [0,0], [1,1]])
training_labels = np.array([[1], [0], [0], [0], [1]])

validation_features = np.array([[0,0], [0,1], [1,0], [1,1]])
validation_labels = np.array([[0], [1], [0], [1]])

test_features = np.array([[0,0], [0,1], [1,0], [1,1]])
test_labels = np.array([[1], [1], [0], [1]])

best = dt.DT_train_binary_best(training_features,training_labels, validation_features, validation_labels)
acc = dt.DT_test_binary(validation_features, validation_labels, best)