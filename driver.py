import numpy as np
import decision_trees as dt

print("Question 1 + 2:")
X = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
Y = np.array([[1], [1], [0]])
max_depth = 2

DT = dt.DT_train_binary(X, Y, max_depth)
test_acc = dt.DT_test_binary(X, Y, DT)
print(test_acc)

training_features = np.array([[0, 1], [0, 0], [1, 0], [0, 0], [1, 1]])
training_labels = np.array([[1], [0], [0], [0], [1]])

validation_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
validation_labels = np.array([[0], [1], [0], [1]])

test_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_labels = np.array([[1], [1], [0], [1]])

best = dt.DT_train_binary_best(training_features, training_labels, validation_features, validation_labels)
acc = dt.DT_test_binary(validation_features, validation_labels, best)

print("Question 3:")
training_set_one = np.array([[0, 1], [0, 0], [1, 0], [0, 0], [1, 1]])
training_labels_one = np.array([[1], [0], [0], [0], [1]])

validation_set_one = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
validation_labels_one = np.array([[0], [1], [0], [1]])

testing_set_one = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
testing_labels_one = np.array([[1], [1], [0], [1]])

training_set_two = np.array(
    [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1],
     [0, 1, 0, 0]])
training_labels_two = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])

validation_set_two = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
validation_labels_two = np.array([[0], [0], [1], [0], [1], [1]])

testing_set_two = np.array(
    [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1],
     [0, 1, 0, 0]])
testing_labels_two = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])

# do stuff here
newDT = dt.DT_train_binary_best(training_set_one, training_labels_one, validation_set_one, validation_labels_one)

# end stuff here
print("Question 4: ")
training_one = np.array(
    [[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1],
     [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])
label_one = np.array([[1], [1], [0], [1], [0], [1], [1], [0]])

sample_1 = np.array([[0, 1, 1, 1, 0, 1, 0]])
sample_2 = np.array([[0, 1, 1, 1, 0, 0, 1]])
sample_3 = np.array([[0, 0, 0, 1, 1, 0, 0]])

first_five = [list(training_one[i]) for i in range(5)]
middle_five = [list(training_one[i]) for i in range(2, 7)]
last_five = [list(training_one[i]) for i in range(3, 8)]
print(first_five)
print(middle_five)
print(last_five)
