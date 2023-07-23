import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('')  # original data

data = np.array(data)  # data into numpy array
m, n = data.shape  # m - rows, n - columns
np.random.shuffle(data)

# each example is a row, tranpose so each example is a column
data_dev = data[0:1000].T  # first 1000 examples. 
Y_dev = data_dev[0]  # header label
X_dev = data_dev[1:n]  # test set

data_train = data[1000:m].T  # rest of examples
Y_train = data_train[0]  # header label
X_train = data_train[1:n]  # training set

