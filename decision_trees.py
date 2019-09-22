import numpy as np
import math


# @params X is test data
# @params Y is label data
# @params max_depth is an integer
# @return


def DT_train_binary(X, Y, max_depth):
    root = Tree()
    X = np.transpose(X)
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
        info_vals = [computeInfoGain(X[i], Y) for i in range(0, len(X))]
        best_gain_index = np.argmax(info_vals)
        # split on best gain index - remove the feature from the feature set
        # adjust the label set accordingly
        data, no, yes = trim_data_sets(best_gain_index, X, Y)
        root.left = DT_train_binary(data, no, max_depth-1)
        root.right = DT_train_binary(data, yes, max_depth-1)
        root.feature = best_gain_index
        print(root.feature)
        print("no errors")
        return root


def trim_data_sets(best_gain, feature_set, labelData):
    no = []
    yes = []
    selector = [x for x in range(len(feature_set)) if x != best_gain]
    data = feature_set[selector, :]
    split_feature = feature_set[best_gain]
    no = [list(labelData[i]) for i in range(0, len(split_feature)) if split_feature[i] == 0]
    yes = [list(labelData[i]) for i in range(0, len(split_feature)) if split_feature[i] == 1]
    return data, no, yes

# Calculate the entropy of the dataset
# Found documentation for the unique function on Stack Overflow link below:
# https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
def entropy(Y):
    unique, counts = np.unique(Y, return_counts=True)
    ent = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(0, len(counts))])
    return ent


# This function computes the information gain of the feature
# @params X a 2d dataset
# @params Y a 2d array representing label data
def computeInfoGain(X, Y):
    total_entropy = entropy(Y)
    uniqueX, countsX = np.unique(X, return_counts=True)
    node_entropy = [(-countsX[i]/np.sum(countsX))*np.log2(countsX[i]/np.sum(countsX)) for i in range(0, len(countsX))]
    labelData = np.array(Y).flatten()
    no = [labelData[i] for i in range(0, len(X)) if X[i] == 0]
    yes = [labelData[i] for i in range(0, len(X)) if X[i] == 1]
    if len(no) == 0:
        infoGain = total_entropy - (len(yes)/np.sum(countsX))*node_entropy[0]
    elif len(yes) == 0:
        infoGain = total_entropy - (len(no)/np.sum(countsX))*node_entropy[0]
    else:
        infoGain = total_entropy - (len(no)/np.sum(countsX))*node_entropy[0] - (len(yes)/np.sum(countsX))*node_entropy[1]
    return infoGain

# This function finds and the most frequent binary value in the dataset passed to it
# @params Y is label data
def most_frequent(Y):
    if len(Y) == 0:
        return 1
    no = 0
    yes = 0
    # print("Y: " + "\n")
    # print(Y)
    # print("\n")
    for i in Y:
        print(i)
        if i == [1]:
            yes += 1
        else:
            no += 1
    if yes > no:
        return 1
    else:
        return 0


# This function will return a leaf node in the form of a list of lists
# @params guess is the DT prediction
def leaf(guess):
    return [guess, [], []]


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
def DT_test_binary(X,Y,DT):
    total_tested = np.size(Y)
    total_correct = 0

    for i in range(0, total_tested):
        # print("start traversal")
        result = traverse(DT,X[i])
        # print("end traversal")
        print("result: " + str(result) + "\n" + "Label: " + str(Y[i]))
        if result == Y[i]:
            total_correct = total_correct + 1

    accuracy = total_correct/total_tested
    print("total correct: " + str(total_correct))
    print("total tested: " + str(total_tested))
    return accuracy







# Node class for Decision Tree Model
# left and right are the children of the node
# Instead of having an isLeaft field, the result field will answer yes (1) or no (0)
#   and be a -1 if the node is not a leaf
# The feature is the index of the feature array, so the model is dependent on the input data being the same type
# as the training data
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.result = -1
        self.feature = None

    def left(self):
        return self.left

    def right(self):
        return self.right

    def setleft(self,node):
        self.left = node

    def setright(self,node):
        self.right = node

    def height(self):
        left_height = 0
        right_height = 0

        if self.left != None:
            left_height = self.height(self.left)
        if self.right != None:
            right_height = self.height(self.right)
        return max(left_height + 1 ,right_height + 1)


def traverse(node,X):
        # print("Feature-split: " + str(node.feature))
        # print("Result: " + str(node.result))

        if node.result != -1:
            return node.result

        elif X[node.feature] is 0:
            return traverse(node.left, X)

        else:
            return traverse(node.right,X)





