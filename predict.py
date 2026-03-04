"""Interactively predict digits using a trained model.

Usage:
    python predict.py
    python predict.py --weights weights/model_weights.npz
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data import get_mnist
from nn import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description="Predict MNIST digits interactively")
    parser.add_argument("--weights", type=str, default="weights/model_weights.npz")
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    print("Loading data and model...")
    x_train, y_train, x_test, y_test = get_mnist()
    net = NeuralNetwork(hidden_size=args.hidden)
    net.load_weights(args.weights)

    test_loss, test_acc = net.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%\n")

    all_images = np.concatenate([x_train, x_test], axis=0)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    total = all_images.shape[0]

    while True:
        try:
            index = int(input(f"Enter an index (0 - {total - 1}), or -1 to quit: "))
            if index == -1:
                break
            if not (0 <= index < total):
                print(f"Index out of range. Please enter 0 - {total - 1}.")
                continue
        except ValueError:
            print("Please enter a valid integer.")
            continue

        img = all_images[index]
        true_label = np.argmax(all_labels[index])
        predicted, probs = net.predict(img)

        plt.imshow(img.reshape(28, 28), cmap="Greys")
        plt.title(f"Predicted: {predicted}  (True: {true_label})  Confidence: {probs[predicted]:.1%}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
