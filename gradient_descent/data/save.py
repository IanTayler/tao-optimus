import numpy as np


def save_numpy(path: str, arr: np.ndarray):
    """Save a numpy array into .npy format."""
    np.save(path, arr)
