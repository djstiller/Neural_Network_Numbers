# Neural Network Numbers

A feedforward neural network implemented **from scratch** using only NumPy to classify MNIST handwritten digits (0-9). No deep learning frameworks — just matrix math and backpropagation.

## How It Works

The network takes a 28x28 grayscale image of a handwritten digit, flattens it into a 784-element vector, passes it through a hidden layer with sigmoid activation, and outputs a probability distribution over the 10 digit classes using softmax.

### Architecture

```text
Input (784) ──> Hidden (128, sigmoid) ──> Output (10, softmax)
```

| Component       | Details                                        |
|-----------------|------------------------------------------------|
| Input layer     | 784 neurons (28x28 pixel images, flattened)    |
| Hidden layer    | 128 neurons, sigmoid activation                |
| Output layer    | 10 neurons (one per digit), softmax activation |
| Loss function   | Cross-entropy                                  |
| Optimizer       | Mini-batch stochastic gradient descent (SGD)   |
| Batch size      | 64 (default)                                   |
| Learning rate   | 0.5 (default)                                  |

### Key Concepts

- **Forward propagation:** Input is multiplied by weights, biases are added, and activation functions are applied layer by layer to produce predictions.
- **Softmax:** Converts raw output scores into probabilities that sum to 1. Used on the output layer for multi-class classification.
- **Cross-entropy loss:** Measures how far the predicted probability distribution is from the true label. Produces well-scaled gradients for faster learning.
- **Backpropagation:** Computes gradients of the loss with respect to each weight by applying the chain rule backwards through the network.
- **Mini-batch SGD:** Instead of updating weights after every single image (slow, noisy) or the entire dataset (memory-heavy), updates are done on small batches of 64 images for a balance of speed and stability.
- **Data shuffling:** Training data is shuffled each epoch to prevent the network from learning the order of samples.

## Requirements

- Python 3.10 or higher
- NumPy
- Matplotlib

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/djstiller/Neural_Network_Numbers.git
   cd Neural_Network_Numbers
   ```

2. Install dependencies:

   ```bash
   pip install numpy matplotlib
   ```

The MNIST dataset (`data/mnist.npz`) is included in the repository.

## Usage

### Training

Run with default settings (10 epochs, lr=0.5, batch size=64, 128 hidden neurons):

```bash
python train.py
```

Customize hyperparameters via command-line arguments:

```bash
python train.py --epochs 20 --lr 0.1 --batch-size 128 --hidden 256
```

| Argument       | Default                      | Description                     |
|----------------|------------------------------|---------------------------------|
| `--epochs`     | 10                           | Number of training epochs       |
| `--lr`         | 0.5                          | Learning rate                   |
| `--batch-size` | 64                           | Mini-batch size                 |
| `--hidden`     | 128                          | Number of hidden layer neurons  |
| `--save`       | `weights/model_weights.npz`  | Path to save trained weights    |

Training output shows both training and test metrics each epoch:

```text
Epoch  1/10  |  Train Loss: 0.3883  Acc: 88.51%  |  Test Loss: 0.2507  Acc: 92.62%
Epoch  2/10  |  Train Loss: 0.2108  Acc: 93.82%  |  Test Loss: 0.1826  Acc: 94.48%
Epoch  3/10  |  Train Loss: 0.1632  Acc: 95.21%  |  Test Loss: 0.1612  Acc: 95.24%
...
```

Trained weights are automatically saved to `weights/model_weights.npz` when training completes.

### Interactive Prediction

After training, run the prediction script to test the model on individual images:

```bash
python predict.py
```

You'll be prompted to enter an image index. The script displays the image with the predicted digit, true label, and confidence score. Enter `-1` to quit.

| Argument     | Default                      | Description                              |
|--------------|------------------------------|------------------------------------------|
| `--weights`  | `weights/model_weights.npz`  | Path to saved weights file               |
| `--hidden`   | 128                          | Hidden layer size (must match training)  |

## Project Structure

```text
Neural_Network_Numbers/
├── data/
│   └── mnist.npz        # MNIST dataset (60k train + 10k test images)
├── weights/
│   └── .gitkeep          # Trained weights saved here (gitignored)
├── data.py               # MNIST data loading with train/test split
├── nn.py                 # NeuralNetwork class (forward, backward, save/load)
├── train.py              # Training script with CLI arguments
├── predict.py            # Interactive digit prediction & visualization
├── pyproject.toml        # Project configuration
├── LICENSE               # MIT license
└── README.md
```

### File Details

- **`data.py`** — Loads MNIST from `data/mnist.npz`, normalizes pixels to [0,1], flattens 28x28 images to 784 vectors, one-hot encodes labels. Returns both training (60,000) and test (10,000) splits.
- **`nn.py`** — Contains the `NeuralNetwork` class with methods: `forward()`, `backward()`, `train_epoch()`, `evaluate()`, `predict()`, `save_weights()`, `load_weights()`. All math is done with NumPy matrix operations.
- **`train.py`** — Entry point for training. Parses CLI arguments, runs the training loop, prints per-epoch train/test metrics, and saves weights on completion.
- **`predict.py`** — Entry point for inference. Loads saved weights, reports test accuracy, then enters an interactive loop where you can visualize predictions on individual images.

## Expected Performance

| Epochs | Approx. Test Accuracy |
|--------|-----------------------|
| 3      | ~95%                  |
| 10     | ~97%                  |
| 20     | ~97.5%                |

Performance depends on the random weight initialization and learning rate. Results may vary slightly between runs.

## License

MIT
