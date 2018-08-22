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
        self.momentum = 0

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

    #def __init__(self, input_neurons, hidden_neurons, output_neurons):
    #    self.hidden_layer = [Neuron(input_neurons) for _ in range(hidden_neurons)]
    #    self.output_layer = [Neuron(hidden_neurons) for _ in range(output_neurons)]

    def forward(self, data):
        self.layer_outputs = list()
        output = data
        self.layer_outputs.append(data)
        for layer in self.layers:
            output = [neuron.forward(output) for neuron in layer]
            self.layer_outputs.append(output)
        return output
        
    #def forward(self, data):
    #    self.hidden_output = [neuron.forward(data) for neuron in self.hidden_layer]
    #    self.output = [neuron.forward(self.hidden_output) for neuron in self.output_layer]
    #    return self.output

    def backpropagate(self, inputs, target):
        #hidden_layer_error = list()

        error_values = [t_val - output for (t_val, output) in zip(target, self.layer_outputs[-1])]
        for i in range(len(self.layers)-1,-1,-1): 
            error_values = self.backpropagate_layer(i, error_values, self.layer_outputs[i])

            

        #error_values = self.backpropagate_layer(self.output_layer, error_values, self.hidden_output)
        #self.backpropagate_layer(self.hidden_layer, error_values, inputs)

        # Output layer
        #for neuron, error in zip(self.output_layer, error_values):
            #gradient = error * sigmoid_prime(neuron.output)
            #for i, input_neuron in enumerate(self.hidden_layer):
                #neuron.weights[i] += Neuron.learning_rate * gradient * input_neuron.output 
                #hidden_layer_error.append(neuron.weights[i]  * gradient)

        # Hidden layer
        #for neuron, error in zip(self.hidden_layer, hidden_layer_error):
            #gradient = error * sigmoid_prime(neuron.output)
            #for i, inp in enumerate(inputs):
                #neuron.weights[i] += Neuron.learning_rate * gradient * inp

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


db = data.Data()
nn = Network((4,7,10,3))

training = db.rows[:100]
testing = db.rows[100:]

num_enocs = 1000

for _ in range(num_enocs):
    random.shuffle(training)
    for row in training:
        nn.forward(row.normalized)
        nn.backpropagate(row.normalized, row.type_vect)
    print(sum([loss(row.type_vect, nn.forward(row.normalized)) for row in db.rows]))

print(nn.forward(db.rows[0].normalized))

num_correct = 0
for row in testing:
    output = nn.forward(row.normalized)
    l = [int(x == max(output)) for x in output]
    if l == row.type_vect:
        num_correct += 1

print(num_correct)
    
