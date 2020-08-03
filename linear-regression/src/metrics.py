import numpy as np

def mean_squared_error(estimates, targets):
    # """
    # Mean squared error measures the average of the square of the errors (the
    # average squared difference between the estimated values and what is
    # estimated. The formula is:
    #
    # MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2
    #
    # Implement this formula here, using numpy and return the computed MSE
    #
    # https://en.wikipedia.org/wiki/Mean_squared_error
    #
    # """test_mean_squared_error
    # test_generate_regression_data
    # test_polynomial_regression
    # test_perceptron
    # test_transform_data

    n = estimates.shape[0]

    total = 0

    for i in range(n):
        total += ((estimates[i] - targets[i])**2)

    mse = (1/n) * total

    return mse
