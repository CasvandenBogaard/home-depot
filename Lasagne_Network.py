import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from feature_extraction.extraction import FeatureExtractor
from sklearn.metrics import mean_squared_error
import pandas as pd

class Network:
    def rmse(self, true, test):
        return mean_squared_error(true, test)**0.5

    def get_target_values(self, train, test):
        y_train = train['relevance'].values
        if 'relevance' in test:
            y_test = test['relevance'].values
            return y_train, y_test
        return y_train

    def loadfile(self):
        print("loading file")
        # use preprocessing.py if you want to stem
        if (os.path.isdir('data/stemmed')):
            df_train = pd.read_csv('data/stemmed/train.csv', encoding="ISO-8859-1")
            df_description = pd.read_csv('data/stemmed/product_descriptions.csv', encoding="ISO-8859-1")
            df_attributes = pd.read_csv('data/stemmed/attributes.csv', encoding="ISO-8859-1")
            df_test = pd.read_csv('data/stemmed/test.csv', encoding="ISO-8859-1")
        else:
            df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
            df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
            df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
            df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
        return df_train, df_description, df_attributes, df_test

    def constructFeatures(self, df_description, df_attributes, df_train, df_test):
        print("constructing features")
        fext = FeatureExtractor(df_description, df_attributes, verbose=True)

        df_train = fext.extractTextualFeatures(df_train)
        df_x_train = fext.extractNumericalFeatures(df_train)

        df_test = fext.extractTextualFeatures(df_test)
        x_test  = fext.extractNumericalFeatures(df_test)
        x_train = np.array(df_x_train)

        y_train = np.array(self.get_target_values(df_train, df_test))
        y_train = y_train.reshape(1,y_train.shape[0])
        return x_train, y_train

    def makeNetwork(self, shape):
        print("make network")
        input_var = T.dmatrix('inputs')
        target_var = T.drow('targets')
        l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)

        # Apply 20% dropout to the input data:
        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

        # Add a fully-connected layer of 9 units, using the linear rectifier, and
        # initializing weights with Glorot's scheme (which is the default anyway):
        l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=10,
                nonlinearity=lasagne.nonlinearities.sigmoid,
                W=lasagne.init.GlorotUniform())

        # We'll now add dropout of 50%:
        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.4)

        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        l_out = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=1,
                nonlinearity=lasagne.nonlinearities.identity)
        return l_out, input_var, target_var


    def getFunctions(self, l_out, input_var, target_var):
        print("make functions")
        network = l_out

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.squared_error(prediction, target_var.T)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.


        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        test_fn = theano.function([input_var], prediction)
        return train_fn, test_fn

    def train_network(self, x_train, y_train, train_fn, test_fn, epochs):
        print("training network..")
        i = 0
        # We iterate over epochs:
        for epoch in range(epochs):
            # In each epoch, we do a full pass over the training data:
            i += 1
            train_err = 0
            train_batches = 0
            print(epoch)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            train_err += train_fn(x_train, y_train)
            train_batches += 1

        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        out = test_fn(x_train)
        out = out.reshape(1,out.shape[0])
        print(y_train.shape)
        print(out.shape)

        print(self.rmse(y_train, out))

    def save_params(self, l_out):
        np.savez('model.npz', *lasagne.layers.get_all_param_values(l_out))
        print(lasagne.layers.get_all_param_values(l_out))

    def load_params(self, l_out):
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(l_out, param_values)

    def train_call(self, epochs=1000):
        df_train, df_description, df_attributes, df_test = self.loadfile()
        x_train, y_train = self.constructFeatures(df_description, df_attributes, df_train, df_test)
        l_out, input_var, target_var = self.makeNetwork(x_train.shape)
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(l_out, param_values)
        train_fn, test_fn = self.getFunctions(l_out, input_var, target_var)
        self.train_network(x_train, y_train, train_fn, test_fn,epochs)
        self.save_params(l_out)

    def predict(self, data):
        l_out, input_var, target_var = self.makeNetwork(data.shape)
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(l_out, param_values)
        train_fn, test_fn = self.getFunctions(l_out, input_var, target_var)
        out = test_fn(data)
        out = out.reshape(1, out.shape[0])
        return out

#net = Network()
#net.train_call()






























