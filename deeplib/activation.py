### Activation functions from NNFS book
import numpy as np

# Activation function
class ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        # Define a forward pass through ReLU
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        # Make a copy of dvalues and save it
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative or zero
        self.dinputs[self.dinputs <= 0] = 0

# Activation function softmax
class Softmax:

    # Forward pass definition
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    # Backward pass
    def backward(self, dvalues):
        # Create an empty array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Compute Jacobian Matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Compute sample wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)