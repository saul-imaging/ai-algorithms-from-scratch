"""AlexNet-style convolutional network blocks implemented with NumPy.

This module keeps the implementation intentionally explicit: convolution,
max-pooling, ReLU, flattening, and dense layers are written directly with
NumPy instead of using a deep-learning framework. The example at the bottom
uses a small input and reduced channel counts so the forward pass is quick.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def conv2d(x: np.ndarray, kernels: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Naive NCHW convolution.

    Args:
        x: Input tensor with shape (batch, channels, height, width).
        kernels: Filter tensor with shape (filters, channels, kh, kw).
        bias: Bias vector with shape (filters,).
        stride: Integer stride.
        padding: Zero-padding applied to height and width.
    """
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, _, height, width = x.shape
    filters, _, kh, kw = kernels.shape
    out_h = (height - kh) // stride + 1
    out_w = (width - kw) // stride + 1
    out = np.empty((batch, filters, out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            window = x[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = np.tensordot(window, kernels, axes=((1, 2, 3), (1, 2, 3))) + bias

    return out


def max_pool2d(x: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    batch, channels, height, width = x.shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    out = np.empty((batch, channels, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            window = x[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            out[:, :, i, j] = window.max(axis=(2, 3))

    return out


class AlexNetStyleNumPy:
    """Small AlexNet-style forward model using only NumPy primitives."""

    def __init__(self, num_classes: int = 2, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.conv1_w = rng.normal(0, 0.05, size=(8, 3, 5, 5)).astype(np.float32)
        self.conv1_b = np.zeros(8, dtype=np.float32)
        self.conv2_w = rng.normal(0, 0.05, size=(16, 8, 3, 3)).astype(np.float32)
        self.conv2_b = np.zeros(16, dtype=np.float32)
        self.fc1_w = rng.normal(0, 0.05, size=(16 * 6 * 6, 64)).astype(np.float32)
        self.fc1_b = np.zeros(64, dtype=np.float32)
        self.fc2_w = rng.normal(0, 0.05, size=(64, num_classes)).astype(np.float32)
        self.fc2_b = np.zeros(num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = relu(conv2d(x, self.conv1_w, self.conv1_b, stride=1, padding=2))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = relu(conv2d(x, self.conv2_w, self.conv2_b, stride=1, padding=1))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)
        x = relu(x @ self.fc1_w + self.fc1_b)
        logits = x @ self.fc2_w + self.fc2_b
        return logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(softmax(self.forward(x)), axis=1)


def demo_forward_pass() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(4, 3, 24, 24)).astype(np.float32)
    model = AlexNetStyleNumPy(num_classes=2)
    logits = model.forward(x)
    probs = softmax(logits)
    print("Logits shape:", logits.shape)
    print("Predicted classes:", np.argmax(probs, axis=1))


if __name__ == "__main__":
    demo_forward_pass()

