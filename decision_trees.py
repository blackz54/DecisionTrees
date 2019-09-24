import numpy as np

# Global to allow features to remain in data, but make sure we don't consider them again for a split
split_features = []

# Algorithm based on sudo code algorithm found on page 13 of the "A Course in Machine Learning" textbook
# @params X The feature data in a numpy array (2D) [ [x,x,x,x], [x,x,x,x],..., [x,x,x,x]]
# @params Y The label data in a numpy array (2D)   [ [y],       [y],     ,..., [y]]
# @params max_depth An integer representing the maximum depth of the tree
def DT_train_binary(X, Y, max_depth):
    global split_features
    root = Tree()
    guess = most_frequent(Y)
    if isUnambiguous(Y):
        root.result = guess
        return root

    elif len(X) == 0:
        root.result = guess
        return root

    elif max_depth == 0:
        root.result = guess
        return root

    else:
        # Using computeInfoGain, we generate a list of information gains for each feature
        info_vals = computeInfoGain(X, Y)
        # Get the index of the best information gain in a list. Corresponds to the feature
        best_gain_index = np.argmax(info_vals)
        # Make sure we never split on this feature again
        split_features.append(best_gain_index)
        # print(info_vals)
        # Split on the best feature
        root.feature = best_gain_index
        # Using trim_data_sets we remove the feature information
        data_no, data_yes, no, yes = trim_data_sets(best_gain_index, X, Y)
        # After removal we continue to recurse down both sides of the tree
        root.left = DT_train_binary(data_no, no, max_depth - 1)
        root.right = DT_train_binary(data_yes, yes, max_depth - 1)
        return root

# This function separates the data set for branching in DT_train_binary
# @params best_gain The feature index the split is being processed for
# @params feature_set The feature data in a numpy array
# @params labelData The label data in a numpy array
# @return data_no Feature data for no side of the split
# @return data_yes Feature data for yes side of the split
# @return no Label data for no side of the split
# @return yes Label data for yes side of the split
def trim_data_sets(best_gain, feature_set, labelData):
    data_no = []
    data_yes = []
    no = []
    yes = []

    # Separate labels for when features is no (0) or yes (1)
    for i in range(len(feature_set)):
        if feature_set[i, best_gain] == 0:
            no.append(labelData[i])
        else:
            yes.append(labelData[i])

    # Separate data set for when features is no (0) or yes (1)
    for i in range(len(feature_set)):
        if feature_set[i, best_gain] == 0:
            data_no.append(feature_set[i])
        else:
            data_yes.append(feature_set[i])

    data_no = np.array(data_no)
    data_yes = np.array(data_yes)
    no = np.array(no)
    yes = np.array(yes)

    return data_no, data_yes, no, yes


# Calculate the entropy of the dataset
# Found documentation for the unique function on Stack Overflow link below:
# https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
# @params Y The label data for entropy calculations
# @return The entropy of the data as a scalar value
def entropy(Y):
    unique, counts = np.unique(Y, return_counts=True)
    ent = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(0, len(counts))])
    return ent


# This function computes the information gain for all remaining splits
# @params X The feature data in a numpy array (2D) [ [x,x,x,x], [x,x,x,x],..., [x,x,x,x]]
# @params Y The label data in a numpy array (2D)   [ [y],       [y],     ,..., [y]]
# @return infoGain A numpy array of the information gain for each feature split
#                   For features that have already been considered, the information gain is reported as -1
def computeInfoGain(X, Y):
    global split_features
    # Calculate total entropy of the set
    total_entropy = entropy(Y)
    infoGain = []

    # Transposing X so we can iterate across the features
    X = X.transpose()

    for i in range(len(X)):
        if i in split_features:
            infoGain.append(-1)
        else:
            # Storing label values (Yes/No) independently for feature_no and feature_yes
            # The first list in label_data contains the amount of no and yes for when the feature is no
            # Where the first index is the amount the label is 0 and the second is the amount the label is 1
            # e.g. label_data = [ [#feature_no_label_no, #feature_no_label_yes],
            #                     [#feature_yes_label_no], [#feature_yes_label_yes]
            #                   ]
            label_data = [[0, 0], [0, 0]]
            label_data = np.array(label_data)
            for j in range(0, len(Y)):
                feature_value = X[i, j]
                label_value = Y[j]
                label_data[feature_value, label_value] = label_data[feature_value, label_value] + label_value

            # Convert label_data to a numpy array to do easier calculations later
            label_data = np.array(label_data)
            # Now we calculate entropy for each set

            # Entropy when no
            feature_no_label_no_prob = 0
            feature_no_label_yes_prob = 0
            entropy_when_no = 0

            if np.sum(label_data[0]) != 0:
                feature_no_label_no_prob = label_data[0, 0] / np.sum(label_data[0])
                feature_no_label_yes_prob = label_data[0, 1] / np.sum(label_data[0])

            # Check if either are zero. We decided in class that 0 * infinity = 0 to avoid log(0) issues
            if feature_no_label_yes_prob == 0 and feature_no_label_no_prob == 0:
                entropy_when_no = 0
            elif feature_no_label_no_prob == 0:
                entropy_when_no = -feature_no_label_yes_prob * np.log(feature_no_label_yes_prob)
            elif feature_no_label_yes_prob == 0:
                entropy_when_no = -feature_no_label_no_prob * np.log(feature_no_label_no_prob)
            else:
                entropy_when_no = -(
                        feature_no_label_no_prob * np.log2(feature_no_label_no_prob) +
                        feature_no_label_yes_prob * np.log2(feature_no_label_yes_prob))

            # Entropy when yes
            feature_yes_label_no_prob = 0
            feature_yes_label_yes_prob = 0
            entropy_when_yes = 0

            if np.sum(label_data[1]) != 0:
                feature_yes_label_no_prob = label_data[1, 0] / np.sum(label_data[1])
                feature_yes_label_yes_prob = label_data[1, 1] / np.sum(label_data[1])

            if feature_yes_label_yes_prob == 0 and feature_yes_label_no_prob == 0:
                entropy_when_yes = 0
            elif feature_yes_label_no_prob == 0:
                entropy_when_yes = -feature_yes_label_yes_prob * np.log2(feature_yes_label_yes_prob)
            elif feature_yes_label_yes_prob == 0:
                entropy_when_yes = -feature_yes_label_no_prob * np.log2(feature_yes_label_no_prob)
            else:
                entropy_when_yes = -((feature_yes_label_yes_prob * np.log2(feature_yes_label_yes_prob)) + (
                        feature_yes_label_no_prob * np.log2(feature_yes_label_no_prob)))

            splitEntropy = total_entropy - (
                    ((np.sum(label_data[0]) / len(Y)) * entropy_when_no) +
                    ((np.sum(label_data[1]) / len(Y)) * entropy_when_yes))
            infoGain.append(splitEntropy)

    # Due to python's weird object-reference, we make sure X gets transposed back
    X = X.transpose()
    # return a numpy array
    infoGain = np.array(infoGain)
    return infoGain


# This function finds and the most frequent binary value in the dataset passed to it
# @params Y is label data
def most_frequent(Y):
    if len(Y) == 0:
        return 1
    no = 0
    yes = 0
    for i in Y:
        if i == [1]:
            yes += 1
        else:
            no += 1
    if yes > no:
        return 1
    else:
        return 0


# This function will test ambiguity of the data passed to it
# Y is label data
def isUnambiguous(Y):
    if len(Y) == 0:
        return True
    ele = Y[0]
    for element in Y:
        if element != ele:
            return False
    return True


# Sends each sample through the model and counts how many it gets correct
# It then divides by the total amount tested to acheive the accuracy
def DT_test_binary(X, Y, DT):
    total_tested = np.size(Y)
    total_correct = 0

    for i in range(0, total_tested):
        # print("start traversal")
        result = traverse(DT, X[i])
        # print("end traversal")
        # print("result: " + str(result) + "\n" + "Label: " + str(Y[i]))
        if result == Y[i]:
            total_correct = total_correct + 1

    accuracy = total_correct / total_tested
    # print("total correct: " + str(total_correct))
    # print("total tested: " + str(total_tested))
    return accuracy


# @param X_train features of training data
# @param Y_train labels of training data
# @param X_val features of validation data
# @param Y_val labels of validation data
def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
    best_model = []
    best_accuracy = 0
    # Max depth is defined by the amount of features in the data set
    max_depth = len(X_train)
    for i in range(0, max_depth):
        model = DT_train_binary(X_train, Y_train, i)
        accuracy = DT_test_binary(X_val, Y_val, model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    return best_model


# Node class for Decision Tree Model
# left and right are the children of the node
# Instead of having an isLeaf field, the result field will answer yes (1) or no (0)
#           and be a -1 if the node is not a leaf
# The feature is the index of the feature array, so the model is dependent on the input data being the same type
# as the training data
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.result = -1
        self.feature = None


def traverse(node, X):
    # print("Feature-split: " + str(node.feature))
    # print("Sample feature value: " + str(X[node.feature]))
    if node.result != -1:
        # print("result: ", node.result)
        return node.result

    elif X[node.feature] == 0:
        # print("Going left")
        return traverse(node.left, X)

    else:
        # print("Going right")
        return traverse(node.right, X)


def DT_make_prediction(x, DT):
    res = 0
