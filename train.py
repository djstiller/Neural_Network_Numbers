"""Train the neural network on MNIST digits.

Usage:
    python train.py
    python train.py --epochs 20 --lr 0.1 --batch-size 128 --hidden 256
"""
import argparse
from data import get_mnist
from nn import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--save", type=str, default="weights/model_weights.npz")
    args = parser.parse_args()

    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = get_mnist()
    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Test samples:     {x_test.shape[0]}")

    net = NeuralNetwork(hidden_size=args.hidden, learning_rate=args.lr)
    print(f"\nNetwork: 784 -> {args.hidden} (sigmoid) -> 10 (softmax)")
    print(f"Learning rate: {args.lr}, Batch size: {args.batch_size}")
    print(f"Training for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        train_loss, train_acc = net.train_epoch(x_train, y_train, batch_size=args.batch_size)
        test_loss, test_acc = net.evaluate(x_test, y_test, batch_size=args.batch_size)
        print(
            f"Epoch {epoch + 1:>2}/{args.epochs}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc * 100:.2f}%  |  "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc * 100:.2f}%"
        )

    net.save_weights(args.save)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
