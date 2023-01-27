import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv").to_numpy()

x_train, y_train = train[:,1:], train[:,0]
x_train.shape


data = np.array(train)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape