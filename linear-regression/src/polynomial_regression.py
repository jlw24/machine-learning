import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.
        
        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c
        
        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf
    
        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np
            
            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

        Args:
            degree (int): Degree of polynomial used to fit the data.
        """

        self.degree = degree
        self.weight = None
    
    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        x = np.array([])

        for each in np.nditer(features):
            for i in range(self.degree+1):
                x = np.append(x, each**i)

        x = np.reshape(x, (-1, self.degree+1))

        x_t = np.transpose(x)

        weight = np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), targets)

        self.weight = weight


    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """

        predictions = np.array([])

        f_array = np.array([])

        final = np.array([])

        for each in np.nditer(features):
            for i in range(self.degree + 1):
                f_array = np.append(f_array, each**i)

        f_array = np.reshape(f_array, (features.shape[0], -1))

        for row in f_array:
            predictions = np.append(predictions, row*self.weight)

        predictions = np.reshape(predictions, (-1, self.degree + 1))

        for row in predictions:
            final = np.append(final, sum(row))

        return final

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)

def PolyCoefficients(x, coeffs):
    o = len(coeffs)
    print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):c
        y += coeffs[i]*x**i
    return y

x = np.linspace(0, 9, 10)
coeffs = [1, 2, 3, 4, 5]
plt.plot(x, PolyCoefficients(x, coeffs))
plt.show()
        """

        x = np.linspace(-1, 1, 100)
        y = self.predict(x)
        plt.scatter(features, targets)
        plt.plot(x, y, "r")
        plt.show()


        #raise NotImplementedError()
