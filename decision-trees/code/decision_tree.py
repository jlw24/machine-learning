import numpy as np
from statistics import mode
from math import log2

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.
        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)

        fullAttributionList = self.attribute_names[:]

        # self, features, targets, self.tree, self.attribute names
        self.tree = self.ID3recursion(features, targets, fullAttributionList, mode(targets))

        #raise NotImplementedError()

    def pickBestAttribute(self, features, attrList, targets):

        infoGainDict = {}
        for attr in attrList:
            infoGainDict[attr] = information_gain(features, self.attribute_names.index(attr), targets)

        # print(infoGainDict)

        currentAttribute = max(infoGainDict, key=infoGainDict.get)

        currentAttributeIndex = self.attribute_names.index(currentAttribute)

        return currentAttribute, currentAttributeIndex

    def ID3recursion(self, features, targets, attrList, mostCommonTarget):

        # if there is no data (no rows), return the most common target
        if features.shape[0] == 0:
            return Tree(value=mostCommonTarget, attribute_name='Leaf, no data')

        # if all data have same class, return first element of target
        elif len(np.unique(targets)) == 1:
            return Tree(value=targets[0], attribute_name='Leaf, same classification  value: {}'.format(targets[0]))

        # if list of attributes is empty
        elif len(attrList) == 0:
            return Tree(value=mostCommonTarget, attribute_name='Leaf, no more attributes')

        else:

            newTree = Tree()

            # Pick best attribute
            currentAttribute, currentAttributeIndex = self.pickBestAttribute(features, attrList, targets)

            newTree.attribute_index = currentAttributeIndex
            newTree.attribute_name = currentAttribute
            newTree.branches = ['zero', 'one']

            fullList = attrList.copy()
            fullList.remove(currentAttribute)

            values = list(np.unique(features[:, currentAttributeIndex]))


            for eachVal in values:

                exampleTargets, exampleFeatures = findNewSet(features,targets,eachVal,currentAttributeIndex)

                if exampleFeatures.shape[0] == 0:
                    if eachVal == float(0):
                        newTree.branches[0] = Tree(value= mode(targets), attribute_name=LEAF)
                    else:
                        newTree.branches[1] = Tree(value=mode(targets), attribute_name=LEAF)

                else:
                    subtree = self.ID3recursion(exampleFeatures, exampleTargets, fullList, mode(exampleTargets))
                    x = 0

                    # if new node is not a leaf
                    if len(subtree.branches) != 0:
                        subtree.value = eachVal

                    if eachVal == float(0):
                        newTree.branches[0] = subtree
                    else:
                        newTree.branches[1] = subtree
                    # print('info about features and targets')
                    # print(exampleTargets)
                    # print(exampleFeatures)
                    # print("info about tree: value, branches, name of tree")
                    # print(subtree.value)
                    # print(subtree.branches)
                    # print(subtree.attribute_name)

            return newTree

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """

        self._check_input(features)

        result = np.array(len(features) * [0])

        # for each row of data
        for idx, row in enumerate(features):
            currTree = self.tree
            result[idx] = self.predictRecur(row, currTree)

        # print(result)
        return result
        # raise NotImplementedError()

    def predictRecur(self, row, ourTree):

        # while not leaf
        while len(ourTree.branches) != 0:

            if row[ourTree.attribute_index] == float(0):
                ourTree = ourTree.branches[0]

            else:
                ourTree = ourTree.branches[1]

        # if it is leaf
        return ourTree.value
    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def findNewSet(features, targets, valToTake, currentAttributeIndex):

    exampleIndex = []
    exampleFeatures = []
    exampleTargets = []

    for rowIdx in range(features.shape[0]):
        if features[rowIdx, currentAttributeIndex] == valToTake:
            exampleIndex.append(rowIdx)

    for eachIdx in exampleIndex:
        exampleFeatures.append(features[eachIdx])
        exampleTargets.append(targets[eachIdx])

    # Example Targets and Features
    exampleTargets = np.array(exampleTargets)
    exampleFeatures = np.array(exampleFeatures)

    return exampleTargets, exampleFeatures

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    data = list(features[:, attribute_index])
    label = {}
    totalyesno = targets.shape[0]
    attributecount = {}
    aveInfoEntropy = 0

    for i in range(len(data)):
        if str(data[i]) in label:
            label[str(data[i])].append(i)
        else:
            label[str(data[i])] = [i]

    total_yes = 0
    total_no = 0

    for i in targets:
        if i == 0:
            total_yes += 1
        else:
            total_no += 1

    yesfrac = (total_yes / totalyesno)
    nofrac = (total_no / totalyesno)

    if yesfrac == 0 or nofrac == 0:
        entropyS = 0
    else:
        entropyS = -((yesfrac) * log2(yesfrac)) - ((nofrac) * log2(nofrac))

    entropyA = {}
    fractionA = {}

    for key in label:
        no = 0
        yes = 0

        attributecount[key] = len(label[key])

        for each in label[key]:
            if targets[each] == 0:
                no += 1
            else:
                yes += 1

        frac = [(yes) / (yes + no), (no) / (yes + no)]

        if frac[0] == 0 or frac[1] == 0:
            entro = 0
        else:
            entro = -(frac[0] * (log2(frac[0]))) - (frac[1] * (log2(frac[1])))

        entropyA[key] = entro
        fractionA[key] = frac

    for key in label:
        aveInfoEntropy += (attributecount[key] / totalyesno) * entropyA[key]

    infoGain = entropyS - aveInfoEntropy

    """print("entropy {}".format(entropyA))
    print("fraction [1, 0] format: {}".format(fractionA))
    print("ave info entropy = {}".format(aveInfoEntropy))
    print(attributecount)
    print("entropyS = {}".format(entropyS))
    print(type(infoGain))
    """
    return infoGain
    #raise NotImplementedError()


if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
