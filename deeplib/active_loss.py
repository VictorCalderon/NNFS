### Activation functions from NNFS book
import numpy as np
from .activation import Softmax
from .loss import CategoricalCrossEntropy

class Softmax_CategoricalCrossEntropy:

    # Creates acgivation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation fnction
        self.activation.forward(inputs)
        # Set output
        self.output = self.activation.output
        # Compute and return loss
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Samples
        samples = len(dvalues)

        # Encode one hot if needed
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
    
        # Copy data to modify it
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples