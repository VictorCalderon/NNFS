### Optimizer functions from NNFS book
import numpy as np

# Optimizer class Stochastic Gradient Descent
class SGD:

    # Init optimizer settings
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

