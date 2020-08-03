import numpy as np
from math import atan
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """

    # transformed_features = np.array([])
    #
    # for row in features:
    #     sample = np.array([row[0], (((row[0]**2)+(row[1]**2))**(1/3))])
    #     transformed_features = np.append(transformed_features, sample)
    #
    # transformed_features = np.reshape(transformed_features, (-1, 2))
    #
    # return transformed_features
    transformed_features = np.array([])
    for row in features:
        r = np.sqrt(row[0]**2 + row[1]**2)
        theta = atan(row[1]/row[0])
        sample = np.array([r, theta])
        transformed_features = np.append(transformed_features, sample)

    transformed_features = np.reshape(transformed_features, (-1, 2))

    return transformed_features

    #raise NotImplementedError


class Perceptron():
    def __init__(self, max_iterations=200):

        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """

        self.max_iterations = max_iterations
        self.weights = None
        self.features = None

    def fit(self, features, targets):

        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)

            This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the
        line are one class and points on the other side are the other class
        """
        """
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end

        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm.

        """

        new = np.ones((features.shape[0], 1))
        self.features = np.append(new, features, 1)
        self.weights = np.zeros(self.features.shape[1])

        j = 0

        while j < self.max_iterations:

            j += 1

            for i in range(self.features.shape[0]):

                gx = np.dot(self.weights, self.features[i])

                if gx > 0:
                    hx = 1
                else:
                    hx = -1

                if hx != targets[i]:
                    self.weights += self.features[i] * (targets[i])

            array1 = np.array([])

            for k in range(self.features.shape[0]):

                gx = sum(self.weights*self.features[k])

                if gx > 0:
                    hx = 1
                else:
                    hx = -1

                array1 = np.append(array1, hx)

            if np.array_equal(targets, array1):
                break

            """   
        new = np.ones((features.shape[0], 1))
        self.features = np.append(new, features, 1)
        self.weights = np.zeros(self.features.shape[1])
        test_target = np.array([])

        j = 0

        while j < self.max_iterations:

            test_target = np.array([])

            for i in range(self.features.shape[0]):
                j += 1
                t = targets[i]
                f = np.reshape(self.features[i], self.weights.shape)
                gx = np.dot(self.weights, f)

                if gx > 0:
                    hx = 1

                else:
                    hx = -1

                if hx != t:
                    addition = self.features[i] * (targets[i] - hx)
                    self.weights += addition

            array1 = np.array([])

            for k in range(self.features.shape[0]):

                gx = sum(self.weights * self.features[k])
                if gx > 0:
                    hx = 1
                else:
                    hx = -1
                array1 = np.append(test_target, hx)
            # check if equal, then break of
            # and not np.array_equal(targets, array1)
"""


    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """

        predictions = np.array([])

        for i in range(features.shape[0]):

            gx = np.sum(features[i] * self.weights[1:]) + self.weights[0]

            if gx > 0:
                predictions = np.append(predictions, 1)
            else:
                predictions = np.append(predictions, -1)

        return predictions

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        raise NotImplementedError()
