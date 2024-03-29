import numpy as np
import copy as copy


# Wrapper function for internal training function
# Algorithm based on sudo code algorithm found on page 13 of the "A Course in Machine Learning" textbook
# @params X The feature data in a numpy array (2D) [ [x,x,x,x], [x,x,x,x],..., [x,x,x,x]]
# @params Y The label data in a numpy array (2D)   [ [y],       [y],     ,..., [y]]
# @params max_depth An integer representing the maximum depth of the tree
def DT_train_binary(X, Y, max_depth):
    model_root = Tree()
    model_root.split_features = []
    getModel(X, Y, max_depth, model_root)
    return model_root


# Internal training function. We use this to set-up the node before being sent down the split, to ensure previously-
#       split features are not considered
# Algorithm based on sudo code algorithm found on page 13 of the "A Course in Machine Learning" textbook
# @params X The feature data in a numpy array (2D) [ [x,x,x,x], [x,x,x,x],..., [x,x,x,x]]
# @params Y The label data in a numpy array (2D)   [ [y],       [y],     ,..., [y]]
# @params max_depth An integer representing the maximum depth of the tree
def getModel(X, Y, max_depth, root):
    guess = most_frequent(Y)
    if isUnambiguous(Y):
        root.result = guess
        return root

    elif len(X) == len(root.split_features):
        root.result = guess
        return root

    elif max_depth == 0:
        root.result = guess
        return root

    else:
        # Using computeInfoGain, we generate a list of information gains for each feature
        info_vals = computeInfoGain(X, Y, root.split_features)
        # print(info_vals)

        # Get the index of the best information gain in a list. Corresponds to the feature
        best_gain_index = np.argmax(info_vals)
        # print(best_gain_index)
        # Write-up specification requires we end if max information gain = 0
        if info_vals[best_gain_index] == 0:
            root.result = guess
            return root

        # Make sure we never split on this feature again
        root.split_features.append(best_gain_index)

        # DEBUG print
        # print(info_vals)

        # Split on the best feature
        root.feature = best_gain_index

        # Using trim_data_sets we remove the feature information
        data_no, data_yes, no, yes = trim_data_sets(best_gain_index, X, Y)

        # After removal we continue to recurse down both sides of the tree

        # Create children nodes
        leftChild = Tree()
        rightChild = Tree()

        # Make sure each gets an independent copy of previously split features
        leftChild.split_features = copy.deepcopy(root.split_features)
        rightChild.split_features = copy.deepcopy(root.split_features)

        # Continue algorithm along each branch
        getModel(data_no, no, max_depth - 1, leftChild)
        getModel(data_yes, yes, max_depth - 1, rightChild)

        # Set children of root
        root.left = leftChild
        root.right = rightChild

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
    feature_set = np.array(feature_set)
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
# @params previous_splits_indices A list of the indices of previous splits
# @return infoGain A numpy array of the information gain for each feature split
#                   For features that have already been considered, the information gain is reported as -1
def computeInfoGain(X, Y, previous_splits_indices):
    # Calculate total entropy of the set
    total_entropy = entropy(Y)
    infoGain = []
    # Transposing X so we can iterate across the features
    X = np.array(X).transpose()

    for i in range(len(X)):
        if i in previous_splits_indices:
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
                label_data[feature_value, label_value] = label_data[feature_value, label_value] + 1


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
    max_depth = len(X_train[0])
    for i in range(0, max_depth):
        model = DT_train_binary(X_train, Y_train, i)
        accuracy = DT_test_binary(X_val, Y_val, model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    # print(best_accuracy)
    return best_model


# Node class for Decision Tree Model
# left and right are the children of the node
# Instead of having an isLeaf field, the result field will answer yes (1) or no (0)
#           and be a -1 if the node is not a leaf
# The feature is the index of the feature array, so the model is dependent on the input data being the same type
# as the training data
# Additionally, split_features is a list of previously split features
class Tree:

    def __init__(self):
        self.left = None
        self.right = None
        self.result = -1
        self.feature = None
        self.split_features = []
        self.average_features = []


def traverse(node, X):
    # print("Feature-split: " + str(node.feature))
    #print("Sample feature value: " + str(X[node.feature]))
    if node.result != -1:
        # print("result: ", node.result)
        return node.result

    elif X[node.feature] == 0:
    #   print("Going left")
        return traverse(node.left, X)

    else:
    #    print("Going right")
        return traverse(node.right, X)

def traverseReal(node, X, feature_avgs):
    # print("FEATURE AVERAGES: ", feature_avgs)
    print("Feature-split: " + str(node.feature))
    print("Sample feature value: " + str(X[node.feature]))
    print("Feature Average: ", feature_avgs[node.feature])
    if node.result != -1:
        print("result: ", node.result)
        return node.result

    elif X[node.feature] < feature_avgs[node.feature]:
        print("Going left")
        return traverseReal(node.left, X, feature_avgs)

    else:
        print("Going right")
        return traverseReal(node.right, X, feature_avgs)


def DT_make_prediction(x, DT):
    res = traverse(DT, x)
    return res

def DT_train_real(X, Y, max_depth):
    X, avgs = transformRealData(X)
    dt = DT_train_binary(X, Y, max_depth)
    dt.average_features = np.array(avgs)
    print(np.array(X))
    return dt

def DT_test_real(X, Y, DT):
    return DT_test_real_helper(X, Y, DT)

def DT_test_real_helper(X, Y, DT):
    total_tested = np.size(Y)
    total_correct = 0
    print(len(X))

    for i in range(0, total_tested):
        # print("start traversal")
        result = traverseReal(DT, X[i], DT.average_features)
        # print("end traversal")
        # print("result: " + str(result) + "\n" + "Label: " + str(Y[i]))
        if result == Y[i]:
            total_correct = total_correct + 1

    accuracy = total_correct / total_tested
    # print("total correct: " + str(total_correct))
    # print("total tested: " + str(total_tested))
    return accuracy

def transformRealData(X):
    avgData = [np.sum(X[:, i])/len(X) for i in range(len(X[0]))]
    X = [[0 if X[j, i] < avgData[i] else 1 for j in range(len(X))] for i in range(len(X[0]))]
    return np.array(X).transpose(), avgData

# @param X_train features of training data
# @param Y_train labels of training data
# @param X_val features of validation data
# @param Y_val labels of validation data
def DT_train_real_best(X_train, Y_train, X_val, Y_val):
    best_model = []
    best_accuracy = 0
    # Max depth is defined by the amount of features in the data set
    max_depth = len(X_train[0])
    for i in range(0, max_depth):
        model = DT_train_real(X_train, Y_train, i)
        accuracy = DT_test_real(X_val, Y_val, model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    # print(best_accuracy)
    return best_model