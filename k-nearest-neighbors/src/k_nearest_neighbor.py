import numpy as np 
from .distances import euclidean_distances, manhattan_distances, cosine_distances


def getNeighbor(distance, n, ignore):

    index = []

    if ignore is False:
        num = n
    else:
        num = n + 1

    for i in range(distance.shape[0]):
        distance[i] = np.argsort(distance[i])

    for row in distance:
        index.append(row[:num])

    index = np.array(index)
    index = np.reshape(index, (distance.shape[0], num))
    x = 0
    return index


def getDistance(type, X, Y):
    if type == 'euclidean':
        return euclidean_distances(X, Y)

    elif type == "manhattan":
        return manhattan_distances(X, Y)

    else:
        return cosine_distances(X, Y)


class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.targets = None

        return

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        HINT: One use case of KNN is for imputation, where the features and the targets 
        are the same. See tests/test_collaborative_filtering for an example of this.
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.features = features
        self.targets = targets
        return


    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """

        distance = getDistance(self.distance_measure, features, self.features)

        neighbors = getNeighbor(distance, self.n_neighbors, ignore_first)

        labels = []

        for i in range(features.shape[0]):

            neighborLabel = np.array([])

            for j in range(neighbors.shape[1]):
                check = self.targets[int(neighbors[i][j])]
                neighborLabel = np.append(neighborLabel, check)

            neighborLabel = neighborLabel.reshape([neighbors.shape[1], self.targets.shape[1]])
            neighborLabel = np.transpose(neighborLabel)

            for line in neighborLabel:
                if self.aggregator == 'mean':
                    labels.append(np.mean(line))
                elif self.aggregator == 'median':
                    labels.append(np.median(line))
                else:
                    findmode, idx, counts = np.unique(line, return_index=True, return_counts=True)
                    index = idx[np.argmax(counts)]
                    findmode = line[index]
                    labels.append(findmode)

        labels = np.array(labels)
        labels = np.reshape(labels, (features.shape[0], self.targets.shape[1]))

        return labels

