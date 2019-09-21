import numpy as np
import math


# @params X is test data
# @params Y is label data
# @params max_depth is an integer
# @return


def DT_train_binary(X, Y, max_depth):
    X = np.transpose(X)
    guess = most_frequent(Y)
    if isUnambiguous(Y):
        return leaf(guess)
    elif len(X) == 0:
        return leaf(guess)
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
