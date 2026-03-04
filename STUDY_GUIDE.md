# Study Guide: Understanding the Neural Network

A structured approach to studying this code, going from foundations to the full picture.

## 1. Start with the data pipeline — `data.py`

Read this file first. It's the simplest and teaches you:

- What MNIST actually is (28x28 grayscale images of digits)
- **Normalization** — why divide by 255? (keeps values small so math stays stable)
- **Flattening** — why turn a 2D image into a 1D vector? (neural nets take flat inputs)
- **One-hot encoding** — why represent label `3` as `[0,0,0,1,0,0,0,0,0,0]`? (so the output layer can produce a probability for *each* digit)

Try this: add `print()` statements to see the shapes and values at each step.

## 2. Understand a single forward pass — `nn.py` `forward()`

Focus *only* on the `forward()` method. Trace it by hand with a single image:

- Matrix multiply `W @ x` — what does this do conceptually? (each hidden neuron computes a weighted sum of all 784 pixels)
- Adding bias `b` — why? (allows the neuron to shift its activation threshold)
- **Sigmoid** — squashes values to [0, 1]. Plot it: `y = 1/(1+e^(-x))`
- **Softmax** — turns raw scores into probabilities. Verify they sum to 1

Tip: run this in a Python shell to see it live:

```python
from data import get_mnist
from nn import NeuralNetwork
import numpy as np

x_train, y_train, _, _ = get_mnist()
net = NeuralNetwork()

# Single image forward pass
img = x_train[0].reshape(784, 1)
h, o = net.forward(img)
print(f"Hidden shape: {h.shape}")   # (128, 1)
print(f"Output shape: {o.shape}")   # (10, 1)
print(f"Probabilities: {o.flatten()}")
print(f"Sum: {o.sum():.4f}")        # Should be 1.0
```

## 3. Understand the loss — why cross-entropy?

The loss tells the network "how wrong you are." In `train_epoch()`:

```python
loss = -np.sum(y_batch * np.log(o + 1e-10))
```

- If the true label is `3` and the network outputs 90% confidence for `3` — low loss
- If it outputs 10% confidence for `3` — high loss
- The `log` makes being *very wrong* much more expensive than being *slightly wrong*

Compare this mentally to MSE (`(output - label)^2`) and think about why log-based loss pushes the network harder to fix confident wrong answers.

## 4. Trace backpropagation — `nn.py` `backward()`

This is the hardest part. Study it in this order:

1. **The key insight**: `delta_o = o - l` — the gradient of cross-entropy + softmax simplifies to just "prediction minus truth." If the network predicted `[0.1, 0.9]` and truth is `[1, 0]`, the error is `[-0.9, 0.9]` — push the first output up, second down.

2. **Weight update**: `grad = delta @ h.T` — the gradient for each weight is "how wrong the output was" times "how active the input to that weight was." If a hidden neuron was very active and contributed to a wrong answer, its weight gets a big correction.

3. **Chain rule backwards**: `delta_h = W.T @ delta_o * (h * (1-h))` — the error propagates back through the weights, scaled by the sigmoid derivative `h*(1-h)`.

4. **Learning rate**: multiplies the gradient — controls how big each step is.

Tip: print the gradients for a single batch to see their magnitude:

```python
# After a forward pass, manually call backward and inspect
print(f"delta_o range: {delta_o.min():.4f} to {delta_o.max():.4f}")
```

## 5. Understand mini-batching — `nn.py` `train_epoch()`

Notice how `x_batch` has shape `(784, batch_size)` not `(784, 1)`. All the same matrix math works — NumPy handles 64 images simultaneously. The `1/batch_size` averaging keeps gradients stable regardless of batch size.

## 6. Watch it learn — `train.py`

Run training with 10 epochs and watch the numbers:

```bash
python train.py
```

Ask yourself:

- Why does train accuracy go up each epoch?
- Why is test accuracy sometimes lower? (generalization gap)
- What happens if you change `--lr` to 0.01? To 5.0? (too slow vs. unstable)

## 7. Experiment

This is where real understanding comes from. Try these one at a time:

- Change hidden size to 20 (like the original) — how much worse is it?
- Change hidden size to 512 — does it keep improving?
- Remove the data shuffling in `train_epoch()` — what happens?
- Replace softmax with sigmoid on the output layer — does accuracy drop?
- Set learning rate to 10.0 — what breaks?

## Recommended background reading order

1. **Linear algebra basics** — matrix multiplication, transpose (needed for all the `@` and `.T` operations)
2. **Derivatives / chain rule** — high school calculus is enough (needed for backprop)
3. **3Blue1Brown's neural network series** on YouTube — outstanding visual intuition for exactly this kind of network
4. **The original Bot Academy video** this code is based on — covers the foundational concepts

## Final tip

Don't try to understand everything at once. Go file by file, method by method, and use `print()` liberally to see what the numbers actually look like at each step.
