'''
<h1>Neural Network</h1>
<p>
Deep neural network library, includes support for training through back propagation
as well as reinforcement learning.
<br> Implemented with a scikit learn like api for ease of use.
</p>
<br><h3>Supported activation functions:</h3>
<p>
    <ul>
        <li>ReLu</li>
        <li>Sigmoid</li>
        <li>Tanh</li>
        <li>Linear</li>
        <li>Softmax</li>
    </ul> 
</p>
'''
import numpy as np
from random import shuffle
from math import floor
from random import shuffle, choices
from copy import deepcopy


class Activation:
    def relu(x, derivative:bool=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
        return np.maximum(0, x)


    def sigmoid(x, derivative:bool=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1.0 / (1.0 + np.exp(-x))


    def tanh(x, derivative:bool=False):
        if derivative:
            return 1 - Activation.tanh(x)**2
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    def linear(x, derivative:bool=False):
        if derivative:
            0
        return x


    def softmax(x, derivative:bool=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis = 0)


    def factory(activation:str):

        if activation == 'relu':
            return Activation.relu
        elif activation == 'softmax':
            return Activation.softmax
        elif activation == 'sigmoid':
            return Activation.sigmoid
        elif activation == 'tanh':
            return Activation.tanh
        elif activation == 'linear':
            return Activation.linear
        else:
            raise ValueError('Activation function not supported')          

class DeepNeuralNetwork:
    """Create a Neural Network
    \nconstructor accepts 2 keyword arguments:
    \nconfig: expects a list definining the hidden layers and output specifications.
    \nexample: 2 hidden ReLu layers with 10 softmax output:
    \n [[16, 'relu'],[16, 'relu'],[10, 'softmax']]
    \nexample: 4 hidden ReLu layers with 2 softmax output:
    \n [[16, 'relu'],[32, 'relu'],[64, 'relu'],[8, 'relu'],[2, 'softmax']]
    \n\ninput: expects the amount of nodes in the input layer as an integer"""

    def __init__(self,
                 config:list = [[16, 'relu'],[16, 'relu'],[10, 'softmax']],
                 input:int = 784):
                 
                 self.config = [[input, 'input']]
                 self.config = self.config + config

                 self.N_LAYERS = len(self.config)

                 self.weights = []
                 self.biases = []
                 self.activations = []

                 
                 for i in range(1, self.N_LAYERS):
                    previous_layer_size = self.config[i-1][0]
                    current_layer_size = self.config[i][0]
                    
                    weight = np.random.randn(current_layer_size, previous_layer_size) * np.sqrt(2. / previous_layer_size)
                    bias = np.zeros(current_layer_size)
                    activation = Activation.factory(self.config[i][1])
                    self.weights.append(weight)
                    self.biases.append(bias)
                    self.activations.append(activation)

    def predict(self, x:list):
        """Return hypothesis for a given input"""
        activation = np.array(x) #A0
        z = [np.zeros(activation.shape)]
        a = [activation]

        preactivation = np.dot(self.weights[0], activation) + self.biases[0] #Z1 = dot(W1, A0)
        activation = self.activations[0](preactivation) #A1 = ReLU(Z1)
        z.append(preactivation)
        a.append(activation)
        
        for i in range(1, self.N_LAYERS-1):
            preactivation = np.dot(self.weights[i], activation)  + self.biases[i]
            activation = self.activations[i](preactivation)
            z.append(preactivation)
            a.append(activation)

        self.z = z
        self.a = a

        return activation


    def backward_pass(self, y_train, output):
        z = self.z
        a = self.a
        w = self.weights
        b = self.biases

        changes_to_w = []
        changes_to_b = []

        error = 2 * (output - y_train) / output.shape[0] * self.activations[-1](z[-1], derivative=True)

        error_w = error
        change_w = np.outer(error_w, a[-2]) 
        changes_to_w.append(change_w)

        error_b = error
        changes_to_b.append(error_b)

        for i in range(self.N_LAYERS-2, 0, -1):

            error_w = np.dot(w[i].T, error_w) * self.activations[i](z[i], derivative=True) 
            changes_to_w.append(np.outer(error_w, a[i-1]))

            error_b = np.sum(error_b) * self.activations[i](z[i], derivative=True)
            changes_to_b.append(error_b)

        changes_to_w.reverse()
        changes_to_b.reverse()

        self._ = changes_to_b

        return changes_to_w, changes_to_b


    def update_parameters(self, changes_to_wb):     
        changes_to_w, changes_to_b = changes_to_wb

        for weight, change in zip(self.weights,changes_to_w):
            weight -= self.l_rate * change

        for bias, change in zip(self.biases, changes_to_b):
            bias -= 2 * self.l_rate * change


    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.predict(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def train(self, x_train, y_train, x_val, y_val, epochs:int=10, l_rate:float=0.001, batch_size:int=20):
        """Train network with back propagation using stochastic gradient descent"""
        self.epochs = epochs
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.batches = floor(len(y_train) / batch_size)
    
        for epoch in range(self.epochs):
            training_data = list(zip(x_train, y_train))
            shuffle(training_data)

            accuracy = self.compute_accuracy(x_val, y_val)
            print(accuracy*100)


    def save(self):
        """return network parameters"""
        return self.weights, self.biases


    def load(self, weights_biases):
        """load network parameters"""
        self.weights, self.biases = weights_biases


    def mutate(self, rate=0.25, scale = 0.1):
        """Randomly select network parameters and change them by a random amount
        \nRate decides what the likely hood is a specific value is changed
        \nScale decides what the magnitude of the change will be if selected"""

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if np.random.rand() <= rate:
                        self.weights[i][j][k] += np.random.normal(0,scale=scale)


            for i in range(len(self.biases)):
                for j in range(len(self.biases[i])):
                    if np.random.rand() <= rate:
                        self.biases[i][j] += np.random.normal(0,scale=scale)


    def crossover(self, spouse):
        """Create a new neural network based on the parameters of two parents.
        \nEach parameter has a 50% probability to be selected from either parent."""
        
        child = deepcopy(spouse)

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if np.random.rand() < 0.5:
                        child.weights[i][j][k] = self.weights[i][j][k]

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                if np.random.rand() < 0.5:
                    child.biases[i][j] = self.biases[i][j]         

        return child


    def __call__(self, x:list):
        return self.predict(x)


class Tools:
    class Genetic:
        def create_population(population_size:int, input_size:int, layer_config:list) -> list:
            '''Return a population of networks'''
            return [DeepNeuralNetwork(input = input_size, config = layer_config) for _ in range(population_size)]


        def crossover(population: list, fitnesses: list) -> DeepNeuralNetwork:
            '''For all members of a population, weighted stochastic selection of two individuals based on fitness.
            \nWeights and Biases are then mixed with a 50% probability to be from either parent
            \nReturns a single new newtwork'''
            network_x, network_y = choices(population, weights=fitnesses, k=2)
            return network_x.crossover(network_y)


        def selection(population: list, fitnesses:list, size=None) -> DeepNeuralNetwork:
            '''Stocastic selection of an individual from the population weighted by fitness'''
            if not size:
                size=len(population)
            return choices(population, weights=fitnesses, k=size)


        def mutate(population: list, rate: float = 0.01, scale: float = 0.1) -> None:
            '''In place stochastic adjustment of weights and biases for all members of a population.
            \nRate indicates the odds that any given weight or bias is updated: 0.01 being 1% change to mutate, 0.1 being 10%
            \nScale indicates the magnitude of the mutation'''

            for network in population:
                network.mutate(rate=rate, scale=scale)
