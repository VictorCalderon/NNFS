### Dense layer class NNFS book
import numpy as np

# Dense layer
class Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize layer size
        self.n_neurons = n_neurons
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    # Forward pass
    def forward(self, inputs):
        # Save a copy of the inputs
        self.inputs = inputs
        # Compute output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backpropagation
    def backward(self, dvalues):
        # Compute inputs and weights derivatives
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    # Repr print
    def __str__(self):
        return f"<Dense Layer> Neurons: {self.n_neurons}."