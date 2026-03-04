# Neural Network Numbers

A feedforward neural network implemented from scratch using NumPy to classify MNIST handwritten digits (0-9).

## Architecture

- **Input layer:** 784 neurons (28x28 pixel images, flattened)
- **Hidden layer:** 128 neurons (sigmoid activation)
- **Output layer:** 10 neurons (softmax activation)
- **Loss function:** Cross-entropy
- **Optimizer:** Mini-batch stochastic gradient descent

## Setup

Requires Python 3.10+ and [Poetry](https://python-poetry.org/).

1. Clone this repository
2. Run `poetry install`

## Usage

### Training

```bash
poetry run python train.py
poetry run python train.py --epochs 20 --lr 0.1 --batch-size 128 --hidden 256
```

### Interactive Prediction

```bash
poetry run python predict.py
```

## Project Structure

| File       | Description                                  |
|------------|----------------------------------------------|
| data.py    | MNIST data loading with train/test split     |
| nn.py      | NeuralNetwork class (forward, backward, I/O) |
| train.py   | Training script with CLI arguments           |
| predict.py | Interactive digit prediction & visualization |

## License

MIT
