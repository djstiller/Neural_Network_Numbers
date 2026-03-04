import numpy as np
import pathlib


def get_mnist():
    """Load MNIST dataset with train/test split.

    Returns:
        (x_train, y_train, x_test, y_test) where:
        - x_train: (60000, 784) float32, pixel values in [0, 1]
        - y_train: (60000, 10) float32, one-hot encoded labels
        - x_test:  (10000, 784) float32
        - y_test:  (10000, 10) float32
    """
    data_path = pathlib.Path(__file__).parent.absolute() / "data" / "mnist.npz"
    with np.load(str(data_path)) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten 28x28 images to 784-dimensional vectors
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # One-hot encode labels
    y_train = np.eye(10, dtype="float32")[y_train]
    y_test = np.eye(10, dtype="float32")[y_test]

    return x_train, y_train, x_test, y_test
