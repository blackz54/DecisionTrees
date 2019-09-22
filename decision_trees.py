import numpy as np
import math




# @params X is feature data
# @params Y is label data
# @params max_depth is an integer
# @return


def DT_train_binary(X, Y, max_depth):
    feature_data = X
    label_data = Y
    X = np.transpose(X)
    guess = most_frequent(Y)
    # Case 1: All labels are the same
    if isUnambiguous(Y):
        return leaf(guess)
    # Case 2: No remaining features
    elif len(X) == 0:
        return leaf(guess)
    # Case 3: Remaining features, determine which to split on
    else:
        info_vals, no, yes = zip(*[computeInfoGain(X[i], Y) for i in range(0, len(X))])
        print(info_vals)
        print(no)
        print(yes)

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
    labelData = Y.flatten()
    no = [labelData[i] for i in range(0, len(X)) if X[i] == 0]
    yes = [labelData[i] for i in range(0, len(X)) if X[i] == 1]
    if len(no) == 0:
        infoGain = total_entropy - (len(yes)/np.sum(countsX))*node_entropy[0]
    elif len(yes) == 0:
        infoGain = total_entropy - (len(no)/np.sum(countsX))*node_entropy[0]
    else:
        infoGain = total_entropy - (len(no)/np.sum(countsX))*node_entropy[0] - (len(yes)/np.sum(countsX))*node_entropy[1]
    return infoGain, no, yes

# This function finds and the most frequent binary value in the dataset passed to it
# @params Y is label data
def most_frequent(Y):
    no = 0
    yes = 0
    for i in Y:
        if i > 0:
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
        result = DT.traverse(X[i])
        if result == Y[i]:
            total_correct = total_correct + 1

    accuracy = total_correct/total_tested
    print(accuracy)
    return accuracy


# The model object will store the tree made during training
# The traverse function will traverse the tree given a sample and output the expected label
class Model(object):
    def __init__(self, tree):
        self.tree = tree

    def traverse(self,X):

        if X[self.tree.result] is not -1:
            return self.tree.result

        elif X[self.tree.feature] is 0:
            return self.traverse(self.tree.left(), X)

        else:
            return self.traverse(self.tree.right(),X)

    def print(self):
        left = None
        right = None
        # for i in range(0, self.tree.height()):
        #     print("height = " + str(i), "Feature split =f"+ str(i), )





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

    def height(self):
        left_height = 0
        right_height = 0

        if self.left != None:
            left_height = self.height(self.left)
        if self.right != None:
            right_height = self.height(self.right)
        return max(left_height + 1 ,right_height + 1)


