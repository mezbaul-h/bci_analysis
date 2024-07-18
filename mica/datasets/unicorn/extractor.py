import math
import pathlib
from typing import List, Tuple

import numpy as np
import pyxdf
import torch

from ...settings import settings
from ..bci42a.transformers import FilterBankTransformer
from ..utils import cache_result


def extract_streams_from_xdf(filename: pathlib.Path, eeg_stream_type: str = "Data", channels: List[int] = None):
    streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=True)

    eeg_stream = None
    marker_stream = None

    for stream in streams:
        stream_info: dict = stream["info"]
        stream_type: str = stream_info["type"][0]

        if stream_type == eeg_stream_type:  # EEG data stream
            eeg_stream = stream
        else:  # Marker stream
            marker_stream = stream

    if not eeg_stream:
        raise Exception("no eeg stream found")

    if not marker_stream:
        raise Exception("no marker stream found")

    sampling_rate: int = math.floor(eeg_stream["info"]["effective_srate"])

    # Extract EEG time series data.
    if channels:
        # Extract selected channels.
        eeg_data = np.array([np.array(item)[channels] for item in eeg_stream["time_series"]])
    else:
        # Extract all channels.
        eeg_data = np.array([item for item in eeg_stream["time_series"]])

    # eeg_signals /= 1e6  # Normalize

    # Convert EEG signal timestamps to relative timestamps.
    eeg_timestamps = eeg_stream["time_stamps"]  # - eeg_stream["time_stamps"][0]

    # Extract marker events.
    marker_data = np.array([item[0] for item in marker_stream["time_series"]])

    # Convert marker timestamps to relative timestamps.
    marker_timestamps = marker_stream["time_stamps"]  # - marker_stream["time_stamps"][0]

    return eeg_data, eeg_timestamps, marker_data, marker_timestamps


TRIAL_START = "trial_start"
CUE_START = "cue_start"
CUE_STOP = "cue_stop"


def get_trial_events(marker_data: np.ndarray, marker_timestamps: np.ndarray):
    """
    returns list containing dictionaries containing {'trial':trial_no, 'start':start, 'stop':stop}
    """
    trials = []

    level = None
    start = None
    trial_no = None

    for mark, ts in zip(marker_data, marker_timestamps):
        m = mark.split()

        if m[0] == TRIAL_START:
            trial_no = m[1]
            start = ts
        elif m[0] == CUE_START:
            # start = ts
            level = int(m[1])
        elif m[0] == CUE_STOP:
            stop = ts
            trials.append(
                {
                    "start": start,
                    "stop": stop,
                    "trial": trial_no,
                    "class_id": level,
                }
            )

    return trials


def split_trials(
    eeg_data: np.ndarray,
    eeg_timestamps: np.ndarray,
    marker_data: np.ndarray,
    marker_timestamps: np.ndarray,
    trial_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    trial_events = get_trial_events(marker_data, marker_timestamps)

    x = []
    y = []

    for t in trial_events:
        eeg_start, eeg_stop = np.searchsorted(eeg_timestamps, [t["start"], t["stop"]])

        if (eeg_stop - 1 - eeg_start) > trial_size:
            y.append(t["class_id"])
            x.append(eeg_data[eeg_start : eeg_start + trial_size, :])

    y = np.array(y)
    x = np.array(x).transpose((0, 2, 1))

    return x, y


class UnicornExtractor:
    """
    hello

    Attributes:
        N_TRIALS: Number of trials in each session
    """

    EVENT_CODES = ["cue_start 1", "cue_start 2"]
    SAMPLING_FREQUENCY = 250
    TRIAL_EPOCH_WINDOW = [0, 7]
    TRAIN_EPOCH_WINDOW = [3, 6]
    TEST_EPOCH_WINDOWS = [
        [2, 5],
        [2.5, 5.5],
        [3, 6],
        [3.5, 6.5],
        [4, 7],
    ]
    N_TRIALS = 50
    CHANNELS = list(range(8))
    SUBJECTS = list(range(1, 21))
    SESSIONS = list(range(1, 3))
    RUNS = list(range(1, 5))

    def __init__(self, data_dir: pathlib.Path) -> None:
        self.data_dir = data_dir
        self.transformer = FilterBankTransformer(
            [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]],
            self.SAMPLING_FREQUENCY,
        )

    @staticmethod
    def compose_filename(subject, session, run):
        return f"sub-P{subject:03d}_ses-S{session:03d}_task-Ronak_run-{run:03d}_eeg.xdf"

    def extract_and_transform(self):
        labeled_train_data = []
        labeled_test_data = []

        for sub_no in self.SUBJECTS:
            run1_file = self.data_dir / self.compose_filename(sub_no, self.SESSIONS[0], self.RUNS[0])
            session = self.SESSIONS[0]

            if not run1_file.exists():
                # Try session 2
                run1_file = self.data_dir / self.compose_filename(sub_no, self.SESSIONS[1], self.RUNS[0])
                session = self.SESSIONS[1]

            if not run1_file.exists():
                # No luck :( continue
                continue

            run2_file = self.data_dir / self.compose_filename(sub_no, session, self.RUNS[1])

            if not run2_file.exists():
                continue

            run3_file = self.data_dir / self.compose_filename(sub_no, session, self.RUNS[2])

            if not run3_file.exists():
                continue

            labeled_train_data.extend(
                self.parse_file(
                    file_path=run1_file,
                )
            )
            labeled_train_data.extend(
                self.parse_file(
                    file_path=run2_file,
                )
            )

            run4_file = self.data_dir / self.compose_filename(sub_no, session, self.RUNS[3])

            if run4_file.exists():
                labeled_train_data.extend(
                    self.parse_file(
                        file_path=run3_file,
                    )
                )
                labeled_test_data.extend(
                    self.parse_file(
                        file_path=run4_file,
                        epoch_windows=self.TEST_EPOCH_WINDOWS,
                    )
                )
            else:
                labeled_test_data.extend(
                    self.parse_file(
                        file_path=run3_file,
                        epoch_windows=self.TEST_EPOCH_WINDOWS,
                    )
                )

            yield sub_no, labeled_train_data, labeled_test_data

    @cache_result(cache_dir=settings.OUTPUT_DIR / "file_extraction_cache")
    def parse_file(self, file_path: pathlib.Path, epoch_windows=None) -> List:
        """
        Parse the unicorn data file and return an epoched data.

        Args:
            file_path
                path to the xdf file.

        Returns:
            data : an EEG structure with following fields:
                x: 3d np array with epoched EEG data : chan x time x trials
                y: 1d np array containing trial labels starting from 0
                s: float, sampling frequency
                c: list of channels - can be list of ints.
        """
        if epoch_windows is None:
            epoch_windows = [self.TRAIN_EPOCH_WINDOW]

        eeg_data, eeg_timestamps, marker_data, marker_timestamps = extract_streams_from_xdf(
            file_path, channels=self.CHANNELS
        )
        epoch_size = self.TRIAL_EPOCH_WINDOW[1] - self.TRIAL_EPOCH_WINDOW[0]
        x, y = split_trials(
            eeg_data, eeg_timestamps, marker_data, marker_timestamps, trial_size=epoch_size * self.SAMPLING_FREQUENCY
        )
        y = y[: self.N_TRIALS]

        # Normalization
        x = x * 1e-3
        y = y - 1

        windowed_data = []

        for epoch_window in epoch_windows:
            # Keep only training time window
            # (trials x channels x time)
            x_sub = x[
                : self.N_TRIALS,
                :,
                math.floor(epoch_window[0] * self.SAMPLING_FREQUENCY) : math.floor(
                    epoch_window[1] * self.SAMPLING_FREQUENCY
                ),
            ]

            # have a check to ensure that all the 288 EEG trials are extracted.
            if x_sub.shape[0] != self.N_TRIALS:
                raise Exception(
                    "Could not extracted all the 50 trials from XDF file: {}. Manually check what is the reason for this".format(
                        file_path
                    )
                )

            labeled_data = []

            for i, label in enumerate(y):
                labeled_data.append(
                    {
                        "data": torch.from_numpy(self.transformer(x_sub[i, :, :]).astype(np.float32)),
                        "label": torch.tensor(label, dtype=torch.long),
                    }
                )

            windowed_data.append(
                {
                    "data": labeled_data,
                    "window": epoch_window,
                }
            )

        return windowed_data
