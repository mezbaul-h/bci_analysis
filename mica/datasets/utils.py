import functools
import hashlib
import math
import pathlib

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def cache_result(cache_dir: pathlib.Path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique filename based on the function arguments
            args_str = "_".join(str(arg) for arg in args[1:])  # Skip self
            kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())

            file_id = hashlib.md5(args_str.encode() + kwargs_str.encode()).hexdigest()

            cache_dir.mkdir(exist_ok=True, parents=True)
            filename = cache_dir / f"{args[0].__class__.__name__}_{file_id}.pt"

            if filename.exists():
                print(f"Loading cached result from {filename}")

                return torch.load(filename)
            else:
                result = func(*args, **kwargs)
                print(f"Caching result to {filename}")
                torch.save(result, filename)

                return result

        return wrapper

    return decorator


def sequential_split(dataset, val_split=0.2):
    # Calculate the split index
    split_idx = math.ceil(len(dataset) * (1 - val_split))

    # Create index lists for train and validation
    train_idx = list(range(split_idx))
    val_idx = list(range(split_idx, len(dataset)))

    # Create Subset objects
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset


def stratified_split(dataset, val_split=0.2, random_state=42):
    # Get the labels from your dataset
    labels = [dataset[i]["label"] for i in range(len(dataset))]

    # Perform stratified split
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=val_split, stratify=labels, random_state=random_state
    )

    # Create Subset objects
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset
