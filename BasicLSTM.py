"""
		This code is written by Roxanne Lee for the purpose of exploring a basic LSTM model with PyTorch
		Code was adapted by some of the references below:
		- Basic LSTM Model https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
		- nn.LSTM from pytorch at https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
		- nn.linear from pytorch at https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
		- pytorch LSTM tutorial at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
		- https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def convert_data(data, look_back=1):
	"""
		Method to convert data values where if timestamps x is the data at time (t),
		then target timestamp y is the data at time (t+look_back)
		Parameters
		----------
		data : numpy.ndarray object
				2D matrix of data sample
		look_back : int (default = 1)
				Period of how many timestamps used to predict the subsequent timestamp
				
		Returns
    -------
		x : numpy.ndarray object
			  Data used for predicting target timestamp values
		y : numpy.ndarray object
				Target timestamp values
	"""
	x, y = [], []
	for i in range(len(data) - look_back - 1):
		a = data[i:(i+look_back), 0]
		x.append(a)
		y.append(data[i+look_back, 0])
	return np.array(x), np.array(y)


class PassengerDataset(Dataset):
	def __init__(self, x, y):
		"""
		Built-in method to initialize an instance of the PassengerDataset class
		Subclass of the PyTorch Dataset class
		Parameters
		----------
		self : Object (class Dataset)
				Reference to the current instance of the class (Dataset)
		x : numpy.ndarray object
				Data samples of timestamps used for prediction
		y : numpy.ndarray object
				Data samples of target timestamp values
		"""
		# convert to torch.FloatTensor with automatic differentiation
		self.y = Variable(torch.tensor(y, dtype=torch.float32))
		self.x = Variable(torch.tensor(x, dtype=torch.float32))
		self.len = self.x.shape[0]

	def __len__(self):
		"""
		Method returns length of dataset
		Parameters
		----------
		self : Object (class Dataset)
				Reference to the current instance of the class (Dataset)

		Returns
    -------
		len : int
				size of Dataset
		"""
		return self.len

	def __getitem__(self, idx):
		"""
		Method returns number of samples in the dataset
		----------
		self : Object (class Dataset)
				reference to the current instance of the class (Dataset)
				
		Returns
    -------
		x[idx] - torch.FloatTensor Object at [idx]
		y[idx] - torch.FloatTensor Object at [idx]
		"""
		return self.x[idx], self.y[idx]


class BasicLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers = 1, output_size = 1):
		"""
		Built-in method to initialize an instance of the BasicLSTM class
		Subclass of the PyTorch nn.Module class
		Parameters
		----------
		self : Object (class nn.Module)
				Reference to the current instance of the class (nn.Module)
		input_size : int
				Size of features in the current input
		hidden_size : int
				Number of features in the hidden state
		num_layers : int (default = 1)
				Number of recurrent layers of stacked LSTMs
		output_size : int (default = 1)
				Size of output sample
		"""
		super(BasicLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_size = output_size
		
		self.lstm = nn.LSTM(num_layers=num_layers, input_size=input_size, 
												hidden_size=hidden_size, batch_first=True)
		
		self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

	def forward(self, x_input):
		"""
		Method computes the prediction at every call
		----------
		self : Object (class nn.Module)
				Reference to the current instance of the class (nn.Module)
		x_input : torch.FloatTensor
				Current input sequence
				
		Returns
    -------
		predictions : torch.FloatTensor
				Output predictions
		"""
		# Initialize hidden state
		h_0 = Variable(torch.zeros(
			self.num_layers, x_input.size(0), self.hidden_size))
		
		# Initialize cell state
		c_0 = Variable(torch.zeros(
			self.num_layers, x_input.size(0), self.hidden_size))
		
		# Propagate input through LSTM
		out, (h_n, c_n) = self.lstm(x_input, (h_0, c_0))

		# Pass LSTM output to the linear layer
		predictions = self.linear(torch.relu(out))

		return predictions


# Preprocessing data
data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(data_url, parse_dates=['Month'])
passenger_data = df.iloc[:,1:2].values.astype(float)

# Normalizing data
mm = MinMaxScaler(feature_range=(0, 1))
passenger_data = mm.fit_transform(passenger_data)

# Train test split
test_size = 12 #using last 12 months as the test data
train_size = len(passenger_data) - test_size

# Convert data
x_data, y_data = convert_data(passenger_data)
x_train, y_train = convert_data(passenger_data[:-test_size])
x_test, y_test = convert_data(passenger_data[-test_size:])

# Obtaining dataset
full_dataset = PassengerDataset(x_data, y_data)
train_dataset = PassengerDataset(x_train, y_train)

# Loading dataset
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 64)

# Training with module
num_epochs = 1500
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
output_size = 1

lstm = BasicLSTM(input_size, hidden_size, num_layers, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
	for i, data in enumerate(train_loader):
		y_prediction = lstm(data[:][0].view(-1, hidden_size, 1)).reshape(-1)

		# Zero the gradients
		optimizer.zero_grad()

		# Compute loss
		loss = loss_function(y_prediction, data[:][1])

		# Perform backward pass
		loss.backward()

		# Perform optimization
		optimizer.step()

	# Print statistics
	if epoch % 100 == 0:
		print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Testing
lstm.eval()
train_predict = lstm(full_dataset[:][0].view(-1 ,hidden_size, 1)).reshape(-1)

# Numpy conversion
data_predict = train_predict.data.numpy()
dataY_plot = full_dataset[:][1].data.numpy()

# Inverse normalization
data_predict_inverse = mm.inverse_transform(data_predict.reshape(-1,1))
dataY_plot_inverse = mm.inverse_transform(dataY_plot.reshape(-1,1))

# Visualization
plt.plot(dataY_plot_inverse, label='actual')
plt.plot(data_predict_inverse, label='predicted')
plt.axvline(x=train_size, c='r', linestyle='--', label='test training split')
plt.autoscale(axis='x', tight=True)
plt.suptitle('LSTM Prediction')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.legend()
plt.show()
