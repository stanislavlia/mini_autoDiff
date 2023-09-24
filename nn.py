from toygrad import Value
import random

class Neuron:
    def __init__(self, n_inputs, activation="tanh"):
        
        self.w = [Value(random.uniform(-1, 1)) for  i in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
        
        
    def __call__(self, x):

        self.z = sum( (wi * xi for wi, xi in zip(self.w, x)) ) + self.b

        if self.activation == "tanh":
            self.a = self.z.tanh()
        if self.activation == "relu":
            self.a = self.z.relu()

        if self.activation == "sigmoid":
            self.a = self.z.sigmoid()
        
        
        return self.a
    
    def parameters(self):
        return self.w + [self.b]
    
    
class Layer:
    def __init__(self, n_inputs, units, activation="tanh"):
        self.neurons = [Neuron(n_inputs, activation=activation) for _ in range(units)]
        
    def __call__(self, inputs):
        self.output = [ni(inputs) for ni in self.neurons]
        
        return self.output[0] if len(self.output) == 1 else self.output
    
    def parameters(self):
        
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
            
        return params
    
    
class MLP:
    def __init__(self, n_inputs, layers_sizes, activations):
        self.sizes = [n_inputs] + layers_sizes
        
        self.layers = [Layer(self.sizes[i], self.sizes[i+1], activations[i]) for i in range(len(layers_sizes))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        params = []
        
        for layer in self.layers:
            params.extend(layer.parameters())
            
        return params
    
        
    
    