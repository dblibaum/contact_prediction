import numpy as np
import random as r
import theano
import theano.tensor as T
import lasagne
import time


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv1DLayer(
        network, num_filters=32, filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    network = lasagne.layers.Conv1DLayer(
        network, num_filters=32, filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network

# def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#     if shuffle:
#         indexes = [r.randint(0, len(inputs) - 1) for _ in range(batchsize)]
#
#     for ...:
#         yield inputs[...], targets[...]

# Load the dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Prepare Theano variables for inputs and targets
input_var = T.imatrix('inputs')
target_var = T.ivector('targets')

network = build_cnn(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
    loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

num_epochs = 100

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    batch_size = 200
    start_time = time.time()

    # Shuffle the dataset each epoch
    indexes = [r.randint(0, len(y_train) - 1) for _ in range(len(y_train))]
    X_train = [X_train[i] for i in indexes]
    y_train = [y_train[i] for i in indexes]

    i = 0
    for batch_num in range(len(y_train)/batch_size):
        inputs = X_train[batch_size*batch_num:batch_size*batch_num*2]
        targets = y_train[batch_size*batch_num:batch_size*batch_num*2]
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
    #     inputs, targets = batch
    #     train_err += train_fn(inputs, targets)
    #     train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch_num in range(len(y_val)/batch_size):
        inputs = X_val[batch_size*batch_num:batch_size*batch_num*2]
        targets = y_val[batch_size*batch_num:batch_size*batch_num*2]
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))