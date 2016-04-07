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



# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None, nr_inputs=0):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, nr_inputs),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 9 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=9,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=3,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=1,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def rmse(true, test):
    return mean_squared_error(true, test)**0.5

def get_target_values(train, test):
    y_train = train['relevance'].values
    if 'relevance' in test:
        y_test = test['relevance'].values
        return y_train, y_test
    return y_train

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

fext = FeatureExtractor(df_description, df_attributes, verbose=True)

df_train = fext.extractTextualFeatures(df_train)
df_x_train = fext.extractNumericalFeatures(df_train)

df_test = fext.extractTextualFeatures(df_test)
x_test  = fext.extractNumericalFeatures(df_test)
x_train = df_x_train

y_train = get_target_values(df_train, df_test)



# Prepare Theano variables for inputs and targets
input_var = T.tensor2()
target_var = T.ivector('targets')

network = build_mlp(input_var, len(x_train.columns))

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.


# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()

# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

i = 0
# We iterate over epochs:
for epoch in range(6):
    # In each epoch, we do a full pass over the training data:
    i += 1
    print(i)
    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in iterate_minibatches(x_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1












