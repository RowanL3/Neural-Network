import math
import random
import data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_prime(x):
    return x * (1.0 - x)

def loss(x,y):
    return sum([(a-b)**2 for (a,b) in zip(x,y)])

class Neuron():
    learning_rate = 0.1

    def __init__(self, input_neurons):
        self.weights = [random.uniform(-1,1) for _ in range(input_neurons)]

    def forward(self, inputs):
        dot = sum([x*y for (x,y) in zip(inputs, self.weights)])
        self.output = sigmoid(dot) 
        return self.output

        
class Network():
    def __init__(self, topology):
        self.topology = topology
        self.layers = list()
        for i in range(1,len(topology)):
            self.layers.append([Neuron(topology[i-1]) for _ in range(topology[i])])

    def forward(self, data):
        self.layer_outputs = [data]
        for layer in self.layers:
            output = [neuron.forward(self.layer_outputs[-1]) for neuron in layer]
            self.layer_outputs.append(output)
        return output
        
    def backpropagate(self, inputs, target):

        error_values = [t_val - output for (t_val, output) in zip(target, self.layer_outputs[-1])]
        for i in range(len(self.layers)-1,-1,-1): 
            error_values = self.backpropagate_layer(i, error_values, self.layer_outputs[i])

    def backpropagate_layer(self, layer, error_values, inputs):
        """Returns the error values for the next layer"""
        next_error_values = list()
        # Output layer
        for neuron, error in zip(self.layers[layer], error_values):
            gradient = error * sigmoid_prime(neuron.output)
            for i, inp in enumerate(inputs):
                neuron.weights[i] += Neuron.learning_rate * gradient * inp
                next_error_values.append(neuron.weights[i]  * gradient)

        return next_error_values
