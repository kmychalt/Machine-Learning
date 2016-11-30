import rbf as rbf
import mlp as mlp
import os
import string
import pylab as pl
import numpy as np

#find the data files in the source directory
os.chdir(os.path.abspath(os.path.dirname(__file__)))
train_data = np.genfromtxt('datatraining.txt', delimiter = ',', skip_header = 1)
test_data = np.genfromtxt('datatest.txt', delimiter = ',', skip_header = 1)
test_data2 = np.genfromtxt('datatest2.txt', delimiter = ',', skip_header = 1)

#splitting of the data into training and testing matrices
trainin = train_data[:1333:,2:7:]
traintgt = train_data[:1333:,7::]
testin = test_data[::2,2:7:]
testtgt = test_data[::2,7::]
test2in = test_data2[:1333:,2:7:]
test2tgt = test_data2[:1333:,7::]

#multilayer perceptron model
print('Multilayer Perceptron Model')
net = mlp.mlp(trainin, traintgt, 2, outtype = 'linear')
net.earlystopping(trainin, traintgt, testin, testtgt, 0.001,1000)
net.confmat(test2in, test2tgt)

#rbf model
print('RBF Model')
net = rbf.rbf(trainin,traintgt,6,0,1)
net.rbftrain(trainin,traintgt,0.01,2000)
net.confmat(testin,testtgt)
