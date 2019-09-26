import numpy as np
import decision_trees as dt
"""
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

print("testing q2 binary best")
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
"""
print("Question 4: ")
training_one = np.array(
    [[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1],
     [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])
label_one = np.array([[1], [1], [0], [1], [0], [1], [1], [0]])

sample_1 = np.array([0, 1, 1, 1, 0, 1, 0])
sample_2 = np.array([0, 1, 1, 1, 0, 0, 1])
sample_3 = np.array([0, 0, 0, 1, 1, 0, 0])

first_five = [training_one[i] for i in range(5)]
label_first = [label_one[i] for i in range(5)]
middle_five = [training_one[i] for i in range(2, 7)]
label_middle = [label_one[i] for i in range(2, 7)]
last_five = [training_one[i] for i in range(3, 8)]
label_last = [label_one[i] for i in range(3, 8)]
"""
first_tree = dt.DT_train_binary(first_five, label_first, 5)
acc = dt.DT_test_binary(first_five, label_first, first_tree)
#print(acc)

middle_tree = dt.DT_train_binary(middle_five, label_middle, 5)
acc = dt.DT_test_binary(middle_five, label_middle, middle_tree)
#print(acc)
"""
last_tree = dt.DT_train_binary(last_five, label_last, 5)
acc = dt.DT_test_binary(last_five, label_last, last_tree)
#print(acc)
"""
result = dt.DT_make_prediction(sample_1, first_tree)
print("result of sample 1 on first tree: " + str(result))
result = dt.DT_make_prediction(sample_2, first_tree)
print("result of sample 2 on first tree: " + str(result))
result = dt.DT_make_prediction(sample_3, first_tree)
print("result of sample 3 on first tree: " + str(result))
"""
"""
result = dt.DT_make_prediction(sample_1, middle_tree)
print("result of sample 1 on middle tree: " + str(result))
result = dt.DT_make_prediction(sample_2, middle_tree)
print("result of sample 2 on middle tree: " + str(result))
result = dt.DT_make_prediction(sample_3, middle_tree)
print("result of sample 3 on middle tree: " + str(result))
"""
"""
result = dt.DT_make_prediction(sample_1, last_tree)
print("result of sample 1 on last tree: " + str(result))
result = dt.DT_make_prediction(sample_2, last_tree)
print("result of sample 2 on last tree: " + str(result))
result = dt.DT_make_prediction(sample_3, last_tree)
print("result of sample 3 on last tree: " + str(result))
"""

X = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2],
              [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6],
              [4.9, 2.4, 3.3, 1]])
Y = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
DT = dt.DT_train_real(X, Y, -1)

acc = dt.DT_test_real(X, Y, DT)
print(acc)
