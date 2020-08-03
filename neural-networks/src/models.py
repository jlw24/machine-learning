import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.
	
	Network architecture:
	- Input layer
	- First hidden layer: fully connected layer of size 128 nodes
	- Second hidden layer: fully connected layer of size 64 nodes
	- Output layer: a linear layer with one node per class (in this case 10)

	Activation function: ReLU for both hidden layers

    """

    def __init__(self):

        super(Digit_Classifier, self).__init__()

        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, input):

        input = F.relu(self.hidden1(input))
        input = F.relu(self.hidden2(input))
        input = self.output(input)

        return input


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):

        super(Dog_Classifier_FC, self).__init__()

        self.hidden1 = nn.Linear(12288, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)


    def forward(self, input):
        input = F.relu(self.hidden1(input))
        input = F.relu(self.hidden2(input))
        input = self.output(input)
        return input


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size[0], stride=stride[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size[1], stride=stride[1])
        self.out = nn.Linear(5408, 10)

    def forward(self, input):

        input = input.permute((0, 3, 1, 2))
        input = F.max_pool2d(F.relu(self.conv1(input)), (2, 2))
        input = F.max_pool2d(F.relu(self.conv2(input)), (2, 2))

        input = input.view(-1, self.flattenme(input))
        input = F.relu(self.out(input))


        return input

    def flattenme(self, input):
        size = input.size()[1:]
        num_feat = 1
        for all in size:
            num_feat *= all
        return num_feat

#       super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# For the 1st convolution layer:
# 16 is the number of outputs i.e. the number of convolution neurons,
# 3 is the number of input channels corresponding to a RGB image
# 5 * 5 is the shape of the convolution kernel
# (Likewise for the 2nd convolution layer)
# For the output layer,
# 5408 is the number of input channels corresponding to the flattened image of the previous convolution layer (13 * 13 * 32 )
# 10 is the number of output channels i.e. the number of linear neurons corresponding to the number of possible digit classifications


class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying 
    synthesized images.

    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=kernel_size[0], stride=stride[0])
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=kernel_size[1], stride=stride[1])
        self.conv3 = nn.Conv2d(4, 8, kernel_size=kernel_size[2], stride=stride[2])
        self.output = nn.Linear(8, 2)
        
    def forward(self, input):

        # input = input.permute((0, 3, 1, 2))
        # input = F.max_pool2d(F.relu(self.conv1(input)), (2, 2))
        # input = F.max_pool2d(F.relu(self.conv2(input)), (2, 2))
        # input = F.max_pool2d(F.relu(self.conv3(input)), (2, 2))
        # input = input.view(-1, self.flattenme(input))
        # input = F.relu(self.output(input))
        #
        # return input


        input = input.permute((0, 3, 1, 2))
        input = F.relu(self.conv1(input))
        input = self.pool(input)
        input = F.relu(self.conv2(input))
        input = self.pool(input)
        input = F.relu(self.conv3(input))
        input = self.pool(input)
        input = input.view(-1, self.flattenme(input))
        input = F.relu(self.output(input))

        return input

    def flattenme(self, input):
        size = input.size()[1:]
        num_feat = 1
        for all in size:
            num_feat *= all
        return num_feat














