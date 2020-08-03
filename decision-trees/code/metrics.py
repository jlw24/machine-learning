import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    for i in range(len(predictions)):

        if actual[i] == True and predictions[i] == True:
            true_positives += 1
        elif actual[i] == False and predictions[i] == False:
            true_negatives += 1
        elif actual[i] == False and predictions[i] == True:
            false_positives += 1
        else:
            false_negatives += 1

    matrix = [[true_negatives, false_positives], [false_negatives, true_positives]]
    matrix = np.array(matrix)

    return matrix

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)

    if (TP + TN + FP + FN) == 0:
        accuracynum = 1

    else:
        accuracynum = (TP + TN) / (TP + TN + FP + FN)

    return accuracynum

    # raise NotImplementedError()

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall


    Precision = TP / (TP + FP)=100/ (100+10) = 0.91
    Recall = TP / (TP + FN) = 100 / (100 + 5) = 0.95
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)

    true_negatives = matrix[0][0]
    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]
    true_positives = matrix[1][1]


    if (false_negatives+true_positives) == 0:
        recall = 0
    else:
        recall = true_positives/(true_positives + false_negatives)

    if (false_positives+true_positives) == 0:
        precision = 0
    else:
        precision = true_positives/(true_positives + false_positives)

    return precision,recall
    #raise NotImplementedError()

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)

    def: Fmeasure = (2 * Recall * Precision) / (Recall + Presision) = (2 * 0.95 * 0.91) / (0.91 + 0.95) = 0.92
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    recall, precision = precision_and_recall(actual, predictions)

    if (recall + precision) == 0:
        f1_measurenum = 0
    else:
        f1_measurenum = (2 * recall * precision) / (recall + precision)

    return f1_measurenum

    #raise NotImplementedError()

