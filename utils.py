from dataclasses import dataclass
import numpy as np


def save_transform(transform: np.ndarray, filename: str):
    assert transform.shape == (4, 4), "Transform must be a 4x4 matrix"
    np.savetxt(filename, transform.flatten())


def load_transform(filename: str) -> np.ndarray:
    transform = np.loadtxt(filename)
    assert transform.shape == (16,), "Transform must be a 4x4 matrix"
    return transform.reshape(4, 4)
