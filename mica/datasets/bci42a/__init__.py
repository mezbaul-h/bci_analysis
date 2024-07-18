from typing import Tuple

from torch.utils.data import DataLoader

from mica.settings import settings

from .dataset import Bci42aDataset
from .extractor import Bci42aExtractor


def iter_datasets(batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    extractor = Bci42aExtractor(data_dir=settings.PACKAGE_ROOT_DIR / ".." / "data" / "bci42a")

    for sub_no, train_data, test_data in extractor.extract_and_transform():
        train_set = Bci42aDataset(train_data)
        test_set = Bci42aDataset(test_data)

        yield sub_no, train_set, test_set
