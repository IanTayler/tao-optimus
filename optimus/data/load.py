"""Utils for reading data from files."""
import pickle

import numpy as np


class UnrecognizedArrayExtensionException(Exception):
    """Raised for unknown file extensions while loading."""


def load_numpy(path: str) -> np.ndarray:
    """Load an array from a file."""
    if path.endswith(".asc"):
        with open(path) as arrayf:
            arr = np.loadtxt(arrayf)
    elif path.endswith(".pkl"):
        with open(path, "rb") as arrayf:
            arr = pickle.load(arrayf)
    elif path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    else:
        raise UnrecognizedArrayExtensionException(path)
    return arr
