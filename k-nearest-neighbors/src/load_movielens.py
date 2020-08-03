import csv
import numpy as np
import os

def load_movielens_data(data_folder_path):
    """
    The MovieLens dataset is contained at data/ml-100k.zip. This function reads the
    unzipped content of the MovieLens dataset into a numpy array. The file to read in
    is called ```data/ml-100k/u.data``` The description of this dataset is:

    u.data -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	          user id | item id | rating | timestamp.
	          The time stamps are unix seconds since 1/1/1970 UTC

    Return a numpy array that has size 943x1682, with each item in the matrix (i, j)
    containing the rating user i had for item j. If a user i has no rating for item j,
    you should put 0 for that entry in the matrix.

    with open(data_path, 'r') as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',')

        #quotechar='|', quoting=csv.QUOTE_MINIMAL

        attribute_names = next(readCSV)

        for row in readCSV:
            targets.append(int(row[-1]))
            features.append(list(map(float, row[:-1])))

        attribute_names = attribute_names[:-1]
        features = np.array(features)
        targets = np.array(targets)

        return features, targets, attribute_names
    Args:
        data_folder_path {str}: Path to MovieLens dataset (given at data/ml-100).
    Returns:
        data {np.ndarray}: Numpy array of size 943x1682, with each item in the array
            containing the rating user i had for item j. If user i did not rate item j,
            the element (i, j) should be 0.
    """
    # This is the path to the file you need to load.
    data_file = os.path.join(data_folder_path, 'u.data')

    data = np.zeros([943, 1682])

    userID = []
    itemID = []
    rating = []

    readfile = open(data_file, 'r')

    for row in readfile:
        Type = row.split('\t')
        userID.append(int(Type[0]))
        itemID.append(int(Type[1]))
        rating.append(int(Type[2]))

    userID = np.array(userID)
    userID = userID.reshape(100000, 1)
    itemID = np.array(itemID)
    itemID = itemID.reshape(100000, 1)
    rating = np.array(rating)
    rating = rating.reshape(100000, 1)

    for i in range(userID.shape[0]):
        data[(userID[i][0])-1][(itemID[i][0])-1] = rating[i]

    return data

    # raise NotImplementedError()