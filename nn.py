import numpy as np
import pathlib


class NeuralNetwork:
    """A simple feedforward neural network with one hidden layer.

    Architecture: Input(784) -> Hidden(hidden_size, sigmoid) -> Output(10, softmax)
    Loss: Cross-entropy
    Optimizer: Mini-batch SGD
    """

    def __init__(self, hidden_size=128, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Weight initialization with small random values
        self.w_i_h = np.random.uniform(-0.5, 0.5, (hidden_size, 784))
        self.w_h_o = np.random.uniform(-0.5, 0.5, (10, hidden_size))
        self.b_i_h = np.zeros((hidden_size, 1))
        self.b_h_o = np.zeros((10, 1))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def softmax(x):
        """Numerically stable softmax. x shape: (10, batch_size)"""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, x):
        """Forward pass. x shape: (784, batch_size)
        Returns (hidden_activations, output_probabilities).
        """
        # Input -> Hidden (sigmoid)
        h_pre = self.b_i_h + self.w_i_h @ x
        h = self.sigmoid(h_pre)

        # Hidden -> Output (softmax)
        o_pre = self.b_h_o + self.w_h_o @ h
        o = self.softmax(o_pre)

        return h, o

    def backward(self, x, h, o, labels):
        """Backward pass with gradient descent update.

        The gradient of cross-entropy loss w.r.t. softmax pre-activations
        simplifies to (output - labels).

        x: (784, batch_size), h: (hidden, batch_size),
        o: (10, batch_size), labels: (10, batch_size)
        """
        batch_size = x.shape[1]

        # Output layer gradients
        delta_o = o - labels
        grad_w_h_o = (1.0 / batch_size) * delta_o @ h.T
        grad_b_h_o = (1.0 / batch_size) * np.sum(delta_o, axis=1, keepdims=True)

        # Hidden layer gradients (sigmoid derivative: h * (1 - h))
        delta_h = self.w_h_o.T @ delta_o * (h * (1.0 - h))
        grad_w_i_h = (1.0 / batch_size) * delta_h @ x.T
        grad_b_i_h = (1.0 / batch_size) * np.sum(delta_h, axis=1, keepdims=True)

        # Update weights and biases
        self.w_h_o -= self.learning_rate * grad_w_h_o
        self.b_h_o -= self.learning_rate * grad_b_h_o
        self.w_i_h -= self.learning_rate * grad_w_i_h
        self.b_i_h -= self.learning_rate * grad_b_i_h

    def train_epoch(self, images, labels, batch_size=64):
        """Train for one epoch using mini-batch SGD.

        images: (N, 784), labels: (N, 10)
        Returns (average_loss, accuracy).
        """
        n_samples = images.shape[0]
        indices = np.random.permutation(n_samples)
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]

        total_loss = 0.0
        nr_correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = images_shuffled[start:end].T  # (784, batch)
            y_batch = labels_shuffled[start:end].T   # (10, batch)
            actual_batch = x_batch.shape[1]

            # Forward
            h, o = self.forward(x_batch)

            # Cross-entropy loss
            total_loss += -np.sum(y_batch * np.log(o + 1e-10))
            nr_correct += np.sum(np.argmax(o, axis=0) == np.argmax(y_batch, axis=0))

            # Backward
            self.backward(x_batch, h, o, y_batch)

        return total_loss / n_samples, nr_correct / n_samples

    def evaluate(self, images, labels, batch_size=64):
        """Evaluate on a dataset without updating weights.

        images: (N, 784), labels: (N, 10)
        Returns (average_loss, accuracy).
        """
        n_samples = images.shape[0]
        total_loss = 0.0
        nr_correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = images[start:end].T
            y_batch = labels[start:end].T

            _, o = self.forward(x_batch)

            total_loss += -np.sum(y_batch * np.log(o + 1e-10))
            nr_correct += np.sum(np.argmax(o, axis=0) == np.argmax(y_batch, axis=0))

        return total_loss / n_samples, nr_correct / n_samples

    def predict(self, image):
        """Predict the digit for a single image.

        image: (784,) flat array
        Returns (predicted_digit, probabilities).
        """
        x = image.reshape(784, 1)
        _, o = self.forward(x)
        probs = o.flatten()
        return int(np.argmax(probs)), probs

    def save_weights(self, filepath="weights/model_weights.npz"):
        """Save weights and biases to a .npz file."""
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), w_i_h=self.w_i_h, w_h_o=self.w_h_o,
                 b_i_h=self.b_i_h, b_h_o=self.b_h_o)
        print(f"Weights saved to {path}")

    def load_weights(self, filepath="weights/model_weights.npz"):
        """Load weights and biases from a .npz file."""
        data = np.load(filepath)
        self.w_i_h = data["w_i_h"]
        self.w_h_o = data["w_h_o"]
        self.b_i_h = data["b_i_h"]
        self.b_h_o = data["b_h_o"]
        self.hidden_size = self.w_i_h.shape[0]
        print(f"Weights loaded from {filepath}")
