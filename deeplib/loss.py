### Loss functions from NNFS book
import numpy as np

# Categorical cross entropy for classification
class Loss:
    def calculate(self, output, y):
        # Compute sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return data loss
        return data_loss

class CategoricalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in batch
        n_samples = len(y_pred)
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Check if hot encoded
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError(f'Invalid input dimension: {len(y_true.shape)}')
        # Return losses
        return -np.log(correct_confidences)
    
    # Backpropagation
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Compute len of label vector
        labels = len(dvalues[0])
        # If labels are sparse, turn them into a one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Compute gradients on inputs
        self.dinputs = -y_true / dvalues
        # Normalize gradients
        self.dinputs = self.dinputs / samples
