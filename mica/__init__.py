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
from .settings import settings
from .utils.drawing import draw_loss_acc_graph
from .utils.torch import EarlyStopping, Trainer

mne.utils.set_log_level(ERROR)


def make_trainer(early_stopping_criterion):
    # Model
    model = FBCNet(
        n_chan=8,
        n_class=2,
        n_bands=9,
        stride_factor=3,
    )

    # Loss function
    criterion = nn.NLLLoss(reduction="sum")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        checkpoint_path="./output/checkpoints",
        device=settings.TORCH_DEVICE,
        early_stopping_criteria=[early_stopping_criterion],
    )

    return trainer


def cv(dataset, batch_size=16, epochs=500, n_splits=5, report_file: Optional[pathlib.Path] = None):
    kf = KFold(n_splits=n_splits, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_sampler = RandomSampler(train_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False)

        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Training
        early_stopping_sub_set = EarlyStopping(monitor="val_acc", patience=200, mode="max", restore_best_weights=False)
        trainer = make_trainer(early_stopping_sub_set)
        trainer.train(train_loader, train_loader, epochs=epochs)

        # Evaluation
        trainer.load_best_params()
        eval_report = trainer.evaluate(val_loader)
        fold_results.append(eval_report)

    fold_results.insert(
        0,
        {
            "acc_avg": np.mean([x["acc"] for x in fold_results]),
            "acc_std": np.std([x["acc"] for x in fold_results]),
        },
    )

    if report_file:
        report_file.write_text(json.dumps(fold_results, indent=4))


def ho(
    full_train_set,
    test_set,
    batch_size=16,
    epochs=500,
    ho_fraction=0.2,
    report_file: Optional[pathlib.Path] = None,
    subject_no: int = None,
):
    train_set, val_set = stratified_split(full_train_set, val_split=ho_fraction)
    # train_set, val_set = sequential_split(full_train_set, val_split=0.25)

    # NOTE: RandomSampler > shuffle=True
    full_train_sampler = RandomSampler(full_train_set)
    full_train_loader = DataLoader(full_train_set, batch_size=batch_size, sampler=full_train_sampler, shuffle=False)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    early_stopping_sub_set = EarlyStopping(monitor="val_acc", patience=200, mode="max", restore_best_weights=False)
    trainer = make_trainer(early_stopping_sub_set)

    # Train on subset of train set
    trainer.train(train_loader, val_loader, epochs=epochs)

    early_stopping_full_set = EarlyStopping(
        monitor="val_loss",
        mode="min",
        threshold=trainer.history["train_loss"][-1],
    )

    draw_loss_acc_graph(
        trainer.history["train_loss"],
        trainer.history["train_acc"],
        trainer.history["val_loss"],
        trainer.history["val_acc"],
        filename=f"loss_acc_eog_{subject_no}.png",
    )

    trainer.history = {k: [] for k in trainer.history.keys()}
    trainer.early_stopping_criteria = [early_stopping_full_set]

    trainer.load_best_params()

    # Train on full train set
    trainer.train(full_train_loader, full_train_loader, epochs=epochs)

    # Load best parameters
    trainer.load_best_params()

    test_loaders = []

    if isinstance(test_set, list):
        for item in test_set:
            test_loaders.append(
                {
                    "loader": DataLoader(item["dataset"], batch_size=batch_size, shuffle=False),
                    "window": item["window"],
                }
            )
    else:
        test_loaders.append(
            {
                "loader": DataLoader(test_set, batch_size=batch_size, shuffle=False),
                "window": None,
            }
        )

    results = [
        {
            "eval_report": trainer.evaluate(full_train_loader),
            "window": "TRAIN_SET_EVAL",
        }
    ]

    for item in test_loaders:
        results.append(
            {
                "eval_report": trainer.evaluate(item["loader"]),
                "window": item["window"],
            }
        )

    if report_file:
        report_file.write_text(json.dumps(results, indent=4))


def main():
    batch_size = 16
    epochs = 1500
    all_test_sets = []
    all_train_sets = []
    sub_nums = []

    for sub_no, full_train_set, test_set in iter_datasets():
        if sub_no not in [1, 2, 3]:
            continue

        all_test_sets.append(test_set)
        all_train_sets.append(full_train_set)
        sub_nums.append(sub_no)

        # cv(
        #     full_train_set,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     n_splits=5,
        #     report_file=settings.PACKAGE_ROOT_DIR / ".." / "output" / f"report_cv_P{sub_no:03d}.json",
        # )
        ho(
            full_train_set,
            test_set,
            batch_size=batch_size,
            epochs=epochs,
            ho_fraction=0.3,
            report_file=settings.PACKAGE_ROOT_DIR / ".." / "output" / f"report_ho_P{sub_no:03d}.json",
            subject_no=sub_no,
        )

        time.sleep(0.5)

    # for train_fold, test_fold in KFold(n_splits=len(sub_nums)).split(sub_nums):
    #     sub_no = sub_nums[test_fold[0]]
    #
    #     print("Subject", sub_no)
    #
    #     test_data = all_test_sets[test_fold[0]]
    #     train_data = [all_train_sets[item] for item in train_fold]
    #
    #     full_train_set = ConcatDataset(train_data)
    #     cv(
    #         full_train_set,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         n_splits=5,
    #         report_file=settings.PACKAGE_ROOT_DIR / ".." / "output" / f"report_inter_cv_P{sub_no:03d}.json",
    #     )
    #     ho(
    #         full_train_set,
    #         test_data,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         ho_fraction=0.3,
    #         report_file=settings.PACKAGE_ROOT_DIR / ".." / "output" / f"report_inter_ho_P{sub_no:03d}.json",
    #     )
    #
    # # clear residues
    shutil.rmtree(settings.PACKAGE_ROOT_DIR / ".." / "output" / "file_extraction_cache", ignore_errors=True)
