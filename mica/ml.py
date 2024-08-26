import json
import pathlib
import shutil
import sys
import time
from logging import ERROR
from typing import Optional

import mne.utils
import numpy as np
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset

# from .datasets.bci42a import iter_datasets
from .datasets.unicorn import iter_datasets
from .datasets.utils import sequential_split, stratified_split
from .logger import logger
from .models import fbcnet, lda, rf, svm, transformer
from .models.fbcnet import FBCNet
from .models.fbcsp import train_model
from .settings import settings
from .utils.torch import EarlyStopping, Trainer

mne.utils.set_log_level(ERROR)


def main():
    # bands = [
    #     (8, 12),
    #     (16, 24),
    # ]
    bands = [
        (item[0], item[1])
        # for item in [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
        for item in [[8, 12], [16, 24]]
    ]
    all_test_sets = []
    all_train_sets = []
    sub_nums = []

    for sub_no, full_train_set, test_set in iter_datasets():
        all_test_sets.append(test_set)
        all_train_sets.append(full_train_set)
        sub_nums.append(sub_no)

        train_model(
            bands,
            250,
            test_set,
            full_train_set,
            plot_feature_space=False,
            report_file_base=settings.PACKAGE_ROOT_DIR / ".." / "output" / "fbcsp" / "within",
            subject_no=sub_no,
        )

        time.sleep(0.25)

    for train_fold, test_fold in KFold(n_splits=len(sub_nums)).split(sub_nums):
        sub_no = sub_nums[test_fold[0]]

        test_data = all_test_sets[test_fold[0]]
        train_data = [all_train_sets[item] for item in train_fold]

        full_train_set = ConcatDataset(train_data)
        train_model(
            bands,
            250,
            test_data,
            full_train_set,
            plot_feature_space=False,
            report_file_base=settings.PACKAGE_ROOT_DIR / ".." / "output" / "fbcsp" / "inter",
            subject_no=sub_no,
        )

        time.sleep(0.25)

    # clear residues
    # shutil.rmtree(settings.PACKAGE_ROOT_DIR / ".." / "output" / "file_extraction_cache", ignore_errors=True)
