import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
	batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):

	train_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	if running_mode == "train":

		train_loader = DataLoader(train_set, batch_size, shuffle=shuffle)


		if valid_set is not None:

			validation_loader = DataLoader(valid_set, len(valid_set))

		train_loss = []
		train_accuracy = []
		valid_loss = []
		valid_acc = []
		epoch = 0
		prev_loss = float("-inf")
		epoch_loss = 0


		while epoch < n_epochs and np.abs(prev_loss - epoch_loss) > stop_thr:

			epoch += 1

			model, epoch_train_loss, epoch_train_accuracy = _train(model, train_loader, train_optimizer)

			train_loss.append(epoch_train_loss)
			train_accuracy.append(epoch_train_accuracy)

			if valid_set is not None:

				prev_loss = epoch_loss
				epoch_loss, val_epoch_acc = _test(model, validation_loader)
				valid_acc.append(val_epoch_acc)
				valid_loss.append(epoch_loss)

		loss_output = {"train": train_loss, "valid": valid_loss}
		accuracy_output = {"train": train_accuracy, "valid": valid_acc}

		return model, loss_output, accuracy_output


			# if valid_set is not None:
			# 	print("epoch: {} Train loss: {:.6f} Valid Loss: {:.6f}".format(epoch, ))

			# for epoch in range(n_epochs):
			#
			# 	model, train_loss, train_accuracy = _train(model, train_loader, train_optimizer)
			#
			# 	test_loss, test_accuracy = _test(model, validation_loader)
			#
			# 	acc.append(test_accuracy)
			#
			# 	if acc[-1] == acc[-2]:
			#
			# 		break

	if running_mode == "test":

		test_loader = DataLoader(test_set, batch_size)

		test_loss, test_accuracy = _test(model, test_loader)

		return test_loss, test_accuracy

"""
	training mode: the model is trained and evaluated on a validation set, if provided. 
				   If no validation set is provided, the training is performed for a fixed 
				   number of epochs. 
				   Otherwise, the model should be evaluted on the validation set 
				   at the end of each epoch and the training should be stopped based on one
				   of these two conditions (whichever happens first): 
				   1. The validation loss stops improving. 
				   2. The maximum number of epochs is reached.
"""

"""
	
	Summary of the operations this function should perform:
	1. Use the DataLoader class to generate trainin, validation, or test data loaders
	2. In the training mode:
	   - define an optimizer (we use SGD in this homework)
	   - call the train function (see below) for a number of epochs untill a stopping
	     criterion is met
	   - call the test function (see below) with the validation data loader at each epoch 
	     if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results
""""""
train_dataset = MyDataset(trainX, trainY)
	valid_dataset = MyDataset(validX, validY)

	model = Basic_Model(weight_init=0.5)

	_, _est_loss, _est_acc = run_model(model,running_mode='train', train_set=train_dataset,
		valid_set = valid_dataset, batch_size=1, learning_rate=1e-3,
		n_epochs=5, shuffle=False)
		
	This function either trains or evaluates a model. 

	training mode: the model is trained and evaluated on a validation set, if provided. 
				   If no validation set is provided, the training is performed for a fixed 
				   number of epochs. 
				   Otherwise, the model should be evaluted on the validation set 
				   at the end of each epoch and the training should be stopped based on one
				   of these two conditions (whichever happens first): 
				   1. The validation loss stops improving. 
				   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset 
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
	learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
    	  The value of each key is a list of loss values. Each loss value is the average
    	  of training/validation loss over one epoch.
    	  If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
    	 The value of each key is a list of accuracies (percentage of correctly classified
    	 samples in the dataset). Each accuracy value is the average of training/validation 
    	 accuracies over one epoch. 
    	 If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 
	
	
	"""


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

	"""
	This function implements ONE EPOCH of training a neural network on a given dataset.
	Example: training the Digit_Classifier on the MNIST dataset


	Inputs:
	model: the neural network to be trained
	data_loader: for loading the netowrk input and targets from the training dataset
	optimizer: the optimiztion method, e.g., SGD 
	device: we run everything on CPU in this homework

	Outputs:
	model: the trained model
	train_loss: average loss value on the entire training dataset
	train_accuracy: average accuracy on the entire training dataset
	"""

	lossnum = 0
	accuracy = 0

	model.train()

	for i, (features, labels) in enumerate(data_loader):

		optimizer.zero_grad()
		features, labels = features.to(device),  labels.to(device)

		outputs = model(features.float())

		loss = F.cross_entropy(outputs, labels.long(), reduction="sum")
		lossnum += loss.item()

		predict = outputs.argmax(dim=1, keepdim=True)

		loss.backward()
		optimizer.step()

		accuracy += predict.eq(labels.view_as(predict)).sum().item()

	train_accuracy = (accuracy / len(data_loader.dataset)) * 100
	lossnum /= len(data_loader.dataset)

	return model, lossnum, train_accuracy



	# lossnum = 0
	# accuracy = 0
	#
	# model.train()
	#
	# for i, (features, labels) in enumerate(data_loader):
	#
	# 	features, labels = features.to(device),  labels.to(device)
	# 	outputs = model(features).float()
	# 	loss = nn.CrossEntropyLoss(outputs, labels)
	# 	lossnum += loss
	#
	# 	optimizer.zero_grad()
	# 	optimizer.step()
	#
	# 	total = labels.size(0)
	# 	_, predicted = torch.argmax(outputs.data, 1)
	# 	correct = (predicted == labels).sum().item()
	# 	accuracy += (correct / total)
	#
	# model = outputs
	# train_loss = lossnum/labels.size(0)
	# train_accuracy = accuracy/labels.size(0)
	#
	# return model, train_loss, train_accuracy


def _test(model, data_loader, device=torch.device('cpu')):
	"""
	This function evaluates a trained neural network on a validation set
	or a testing set. 

	Inputs:
	model: trained neural network
	data_loader: for loading the netowrk input and targets from the validation or testing dataset
	device: we run everything on CPU in this homework

	Output:
	test_loss: average loss value on the entire validation or testing dataset 
	test_accuracy: percentage of correctly classified samples in the validation or testing dataset
	"""

	model.eval()
	correct = 0
	test_loss = 0

	with torch.no_grad():

		for images, labels in data_loader:

			images, labels = images.to(device), labels.to(device)
			outputs = model(images.float())
			test_loss += F.cross_entropy(outputs, labels.long(), reduction='sum').item()
			predict = outputs.argmax(dim=1, keepdim=True)
			correct += predict.eq(labels.view_as(predict)).sum().item()

	test_accuracy = (correct / len(data_loader.dataset)) * 100
	test_loss /= len(data_loader.dataset)

	return test_loss, test_accuracy

	# model.eval()
	#
	# with torch.no_grad():
	#
	# 	correct = 0
	# 	total = 0
	#
	# 	for images, labels in data_loader:
	#
	# 		images, labels = images.to(device), labels.to(device)
	# 		outputs = model(images.float())
	# 		_, predicted = torch.argmax(outputs.data, 1)
	# 		total += labels.size(0)
	# 		correct += (predicted == labels).sum().item()
	#
	# test_accuracy = (correct / total) * 100
	#
	# return test_loss, test_accuracy






