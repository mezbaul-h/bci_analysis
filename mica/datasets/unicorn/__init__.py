from typing import Tuple

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mica.settings import settings

from .dataset import UnicornDataset
from .extractor import UnicornExtractor
from .utils import sequential_split, stratified_split


def iter_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    extractor = UnicornExtractor(data_dir=settings.PACKAGE_ROOT_DIR / ".." / "data" / "unicorn")

    for sub_no, train_data, test_data in extractor.extract_and_transform():
        train_set = UnicornDataset(train_data[0]["data"])
        test_sets = []

        for item in test_data:
            test_sets.append(
                {
                    "dataset": UnicornDataset(item["data"]),
                    "window": item["window"],
                }
            )

        yield sub_no, train_set, test_sets
