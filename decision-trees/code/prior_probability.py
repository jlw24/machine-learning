import numpy as np
from statistics import mode

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.

        This is a super basic "machine learner" and not something more complicated.
        It's supposed to just be an intro to passing some actual test cases.
        To implement it, you can simply count the number of points that belong
        to each class and return the more common class.
        from collections import Counter
            a = [1,2,3,1,2,1,1,1,3,2,2,1]
            b = Counter(a)
            print b.most_common(1)
        """
        self.most_common_class = mode(targets)


    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.

        Return:
            Return a separate np array with the predictions for each data point
        """
        prediction = np.array(data.shape[0]*[0])

        for i in range(len(data)):
            prediction[i] = self.most_common_class

        return prediction

        #raise NotImplementedError()