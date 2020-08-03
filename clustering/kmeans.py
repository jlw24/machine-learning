import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        python -W ignore -m pytest -s

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None
        self.assignments = np.array([])

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)

        """

        #initialize means
        self.means = np.random.permutation(features)[:self.n_clusters]

        # set convergence condition
        converged = False

        while not converged:

            old = self.assignments

            distances = np.array([[(np.linalg.norm(i - j)) ** 2 for j in self.means] for i in features])

            self.assignments = np.array([np.argmin(i) for i in distances])

            # for each mean, find the count and the numerator

            for i in range(self.means.shape[0]):
                find = np.where(self.assignments == i, 1, 0)
                count = sum(find)
                idx = np.argwhere(find == 1)
                idx = idx[:, 0]
                feat_dim = features[idx]
                numerator = features[idx].sum(axis=0)
                self.means[i] = numerator / count

            if np.array_equal(old, self.assignments):
                converged = True
        #raise NotImplementedError()


        # # get distances
        # distance = np.array([[(np.linalg.norm(i - j))**2 for j in self.means] for i in features])
        #
        # # assign which mean closest to depending on distance
        # self.assignments = np.array([[np.argmin(j)] for j in distance])
        #
        # converged = False
        #
        # while not converged:
        #
        #     old_labels = self.assignments
        #
        #     for i in range(self.n_clusters):
        #
        #         # find which ones are in that cluster
        #         find = np.where(self.assignments == i, 1, 0)
        #         # sum up the numbers for denominator
        #         denominator = np.sum(find)
        #         # get indexes where examples == mean class
        #         idx = np.argwhere(find == 1)
        #         # only need first index to get row
        #         idx = idx[:, 0]
        #         numerator = features.sum(axis=0)
        #
        #         self.means[i] = numerator/denominator
        #
        #
        #
        #     # get distances
        #     distance = (np.array([[(np.linalg.norm(i - j)) ** 2 for j in self.means] for i in features]))
        #
        #     # assign which mean closest to depending on distance
        #     self.assignments = np.array([[np.argmin(j)] for j in distance])
        #
        #     if np.array_equal(old_labels, self.assignments):
        #         converged = True
        #         print(self.means)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """

        distances = np.array([[(np.linalg.norm(i - j)) ** 2 for j in self.means] for i in features])
        predictions = np.array([np.argmin(i) for i in distances])

        return predictions