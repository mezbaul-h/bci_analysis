import pathlib
from typing import Any, Dict, List

import mne
import numpy as np
import torch
from scipy.io import loadmat

from ...settings import settings
from ..utils import cache_result
from .transformers import FilterBankTransformer


class Bci42aExtractor:
    EVENT_CODES = ["768"]  # start of the trial at t=0
    SAMPLING_FREQUENCY = 250
    EPOCH_OFFSET = 2
    SUBJECTS = list(range(1, 2))

    def __init__(self, data_dir: pathlib.Path) -> None:
        self.data_dir = data_dir
        self.transformer = FilterBankTransformer(
            [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]],
            self.SAMPLING_FREQUENCY,
        )

    def extract_and_transform(self):
        labeled_train_data = []
        labeled_test_data = []

        for sub_no in self.SUBJECTS:
            labeled_train_data.extend(
                self.parse_file(
                    data_path=self.data_dir / f"A{sub_no:02d}T.gdf", label_path=self.data_dir / f"A{sub_no:02d}T.mat"
                )
            )
            labeled_test_data.extend(
                self.parse_file(
                    data_path=self.data_dir / f"A{sub_no:02d}E.gdf", label_path=self.data_dir / f"A{sub_no:02d}E.mat"
                )
            )

            yield sub_no, labeled_train_data, labeled_test_data

    @cache_result(cache_dir=settings.OUTPUT_DIR / "file_extraction_cache")
    def parse_file(
        self, data_path: pathlib.Path, label_path: pathlib.Path, epoch_window=[0, 4], channels=list(range(22))
    ) -> List[Dict[str, Any]]:
        """
        Parse the bci42a data file and return an epoched data.

        Parameters
        ----------
        data_path
            path to the gdf file.
        label_path
            path to the labels mat file.
        epoch_window : list, optional
            time segment to extract in seconds. The default is [0,4].
        channels  : list : channels to select from the data.

        Returns
        -------
        data : an EEG structure with following fields:
            x: 3d np array with epoched EEG data : chan x time x trials
            y: 1d np array containing trial labels starting from 0
            s: float, sampling frequency
            c: list of channels - can be list of ints.
        """
        # Load the gdf file using MNE.
        raw_gdf = mne.io.read_raw_gdf(data_path, preload=True)

        # Convert annotations to events.
        events, event_ids = mne.events_from_annotations(raw_gdf)

        # Create epochs.
        epochs = mne.Epochs(
            raw_gdf,
            events,
            event_id=event_ids[self.EVENT_CODES[0]],
            tmin=epoch_window[0] + self.EPOCH_OFFSET,
            tmax=epoch_window[1] + self.EPOCH_OFFSET,
            baseline=None,
            picks=channels,
            preload=True,
        )

        # MNE returns the end time point as well, so remove that. (trials x channels x time)
        x = epochs.get_data()[:, :, :-1] * 1e6
        # print(x.shape)

        # Extract data as 3D NumPy array (channels x time x trials)
        # x = x.transpose(1, 2, 0) * 1e6

        # (trials x channels x time)
        # have a check to ensure that all the 288 EEG trials are extracted.
        if x.shape[0] != 288:
            raise Exception(
                "Could not extracted all the 288 trials from GDF file: {}. Manually check what is the reason for this".format(
                    data_path
                )
            )

        # Load the labels.
        y = loadmat(str(label_path))["classlabel"].squeeze()

        # Change the labels from [1-4] to [0-3].
        y = y - 1

        labeled_data = []

        for i, label in enumerate(y):
            labeled_data.append(
                {
                    "data": torch.from_numpy(self.transformer(x[i, :, :]).astype(np.float32)),
                    "label": torch.tensor(label, dtype=torch.long),
                }
            )

        return labeled_data
