"""
network.py
~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# Local libraries:
from data import vectorized_result

class Network(object):

    def __init__(self, sizes, weight_init='gaus_random', _seed=None, binary=False):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes

        if weight_init == 'gaus_random':
            if _seed is not None:
                np.random.seed(_seed)
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        elif weight_init == 'zeros':
            self.biases = [np.zeros(y,1) for y in sizes[1:]]
            self.weights = [np.zeros(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        elif weight_init == 'symmetry':
            self.biases = []
            self.weights = []
            pass
            exit()

        self.convergenceData = []
        self.loss = []
        self.binary = binary
        self.convergenceTrain = []

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            epoch_loss = 0
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, eta)
                epoch_loss += loss
            self.loss.append(epoch_loss/len(mini_batches))

            if test_data:
                if self.binary:
                    correct = self.evaluate_binary(test_data)
                else:
                    correct = self.evaluate(test_data)
                print ('Epoch {0}: {1} / {2}'.format(
                    j, correct, n_test), end='')
                self.convergenceData.append(correct)
                train_correct = self.evaluate(training_data)
                self.convergenceTrain.append(train_correct/len(training_data))
                print("  Training: %i/%i"%(train_correct, len(training_data)))
            else:
                print ("Epoch {0} complete".format(j))
        
        if test_data: return correct


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        batch_loss = 0
        for x, y in mini_batch:
            (delta_nabla_b, delta_nabla_w) , loss = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            batch_loss += loss

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        return batch_loss/len(mini_batch)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        
        # backward pass for output layer only:
        _loss = []
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        _loss.append(delta)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]) )

        # backward pass for hidden + input layer(s)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot( np.transpose(self.weights[-l+1]), delta) * sp
            _loss.append(delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]) )

        loss = 0; loss_len = 0
        for l in _loss:
            l = l.flatten()
            l = np.square(l)
            loss += np.sum( l )
            loss_len += len(l)
        loss = np.sqrt(loss)
        loss = loss/loss_len

        return (nabla_b, nabla_w), loss

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [ (np.argmax(self.feedforward(x)), (np.argmax(y)) ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_binary(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        correct = 0
        for (x,y) in test_data:
            result = self.feedforward(x).flatten() # assuming binary default is 0 else 1
            if result >= 0.5:
                # is class 1:
                if y[0][0] == 1:
                    correct += 1
                test_results.append((1,y))
            else:
                # is class 0:
                if y[0][0] == 0:
                    correct += 1
                test_results.append((0,y))

        return correct
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    # Used for XOR:
    def graphBoundary(self, xy, num_classes, network_layout, out_file):
        fig = plt.figure()
        x_ranges = [-10,110]
        y_ranges = [-10,110]
        colors = ['#76c1f5', '#fab77c']
        
        for _x in range(x_ranges[0], x_ranges[1],2):
            for _y in range(y_ranges[0], y_ranges[1],2):
                x = _x/100
                y = _y/100
                _input = [[x], [y]]
                print(_input)

                probs = self.feedforward(_input)
                
                if self.binary:
                    if probs.flatten()[0] >= 0.5:
                        _class = 1
                    else:
                        _class = 0
                else:
                    _class = np.argmax(probs)
                
                if _class == 0: plt.scatter(x,y,c=colors[_class])
                else: plt.scatter(x,y,c=colors[_class])
                
                #XOR: [[0,0], [1,0], [0,1], [1,1]] [0,1,1,0]
                plt.scatter([0,1], [0,1], c='blue')
                plt.scatter([0,1], [1,0], c='red')
                
                plt.xlim([x_ranges[0]/100, x_ranges[1]/100])
                plt.ylim([y_ranges[0]/100, y_ranges[1]/100])
                
                plt.title("Boundaries of MLP %s"%(str(network_layout)))
                plt.plot()
        plt.savefig(out_file)

    def graphConvergence(self, out_file):
        fig = plt.figure()
        x = [i for i in range(len(self.loss))]
        y = self.loss
        plt.title("Loss Value - Network %s"%(str(self.sizes)))
        plt.ylabel("MSE Value")
        plt.xlabel("Epoch")
        plt.plot(x,y)
        plt.savefig(out_file)

    def graphTrain(self, out_file):
        fig = plt.figure()
        x = [i for i in range(len(self.convergenceTrain))]
        y = self.convergenceTrain
        plt.title("Training Accuracy - Network %s"%(str(self.sizes)))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.ylim([0,1])
        plt.plot(x,y)
        plt.savefig(out_file)

    def graphAccuracy(self, out_file, data_len):
        fig = plt.figure()
        x = [i for i in range(len(self.convergenceData))]
        y = [i/data_len for i in self.convergenceData]
        plt.plot(x,y)
        plt.title("Accuracy - %s"%(str(self.sizes)))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim([0,1])
        plt.savefig(out_file)
        

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return np.array(1.0/(1.0+np.exp(-z)))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return np.array(sigmoid(z)*(1-sigmoid(z)))
