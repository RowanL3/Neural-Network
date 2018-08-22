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
    learning_rate = 0.015
    momentum_loss = 0.03

    def __init__(self, input_neurons):
        self.weights = [random.uniform(-1,1) for _ in range(input_neurons)]
        self.momentum = [0 for _ in range(input_neurons)]

    def forward(self, inputs):
        dot = sum([x*y for (x,y) in zip(inputs, self.weights)])
        self.output = sigmoid(dot) 
        return self.output

    def backpropagate(self, inputs, error):
        error_values = list()
        gradient = error * sigmoid_prime(self.output)
        for i, inp in enumerate(inputs):
            self.nudge_weight(i, gradient * inp)
            error_values.append(self.weights[i]  * gradient)

        return error_values
    
    def nudge_weight(self, weight, amount):
        change = amount * Neuron.learning_rate
        self.momentum[weight] += change
        self.momentum[weight] *= (1 - Neuron.momentum_loss)
        self.weights[weight] += change + self.momentum[weight] 

class Network():
    def __init__(self, topology):
        self.topology = topology
        self.layers = list()
        for i in range(1,len(topology)):
            self.layers.append([Neuron(topology[i-1]) for _ in range(topology[i])])

    def forward(self, data):
        output = data
        for layer in self.layers:
            output = [neuron.forward(output) for neuron in layer]
        return output
        
    def backpropagate(self, data, output, target):
        error_values = [tval - output for (tval, output) in zip(target, output)]
        for i in range(len(self.layers)-1,0,-1): 
            layer_output = [neuron.output for neuron in self.layers[i-1]]
            error_values = self.backpropagate_layer(i, error_values, layer_output)
        self.backpropagate_layer(0, error_values, data)

    def backpropagate_layer(self, layer, error_values, inputs):
        next_errors = list()
        for neuron, error in zip(self.layers[layer], error_values):
            bp_error = neuron.backpropagate(inputs,error)
            if not next_errors:
                next_errors = bp_error
            else:
                next_errors = [a+b for a,b in zip(next_errors,bp_error)]

        return next_errors
