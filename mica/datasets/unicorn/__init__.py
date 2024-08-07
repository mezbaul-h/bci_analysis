from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mica.settings import settings

from .dataset import UnicornDataset
from .extractor import UnicornExtractor


def iter_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    ext = UnicornExtractor(data_dir=settings.PACKAGE_ROOT_DIR / ".." / "data" / "unicorn")

    for sub_no, train_data, test_data in ext.extract_and_transform():
        train_set = UnicornDataset(np.array([item["data"] for item in train_data]).flatten())
        test_sets = []

        for item in test_data:
            test_sets.append(
                {
                    "dataset": UnicornDataset(item["data"]),
                    "window": item["window"],
                }
            )

        yield sub_no, train_set, test_sets
