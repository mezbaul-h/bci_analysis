import mne
import numpy as np


def make_raw(raw_data: np.ndarray, sfreq: int) -> mne.io.RawArray:
    """

    Args:
         raw_data: Raw data of shape (n_channels, n_times).
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    montage8 = montage.copy()

    # Channels in our dataset
    channels = ["Fz", "C3", "P3", "FCz", "C4", "A1", "A2", "P4"]

    indices = [i for (i, channel) in enumerate(montage.ch_names) if channel in channels]

    # Keep only the desired channels
    montage8.ch_names = [montage.ch_names[idx] for idx in indices]
    my_chans_info = [montage.dig[idx + 3] for idx in indices]

    # Keep the first three rows as they are the fiducial points information
    montage8.dig = montage.dig[0:3] + my_chans_info

    # create info object to creating mne.raw object
    info = mne.create_info(ch_names=montage8.ch_names, sfreq=sfreq, ch_types="eeg", verbose=None)

    # create the raw.mne object from the numpy array of the data
    raw = mne.io.RawArray(raw_data, info=info, first_samp=0)
    raw.set_montage(montage8)

    return raw


def pre_preprocess(raw: mne.io.RawArray) -> mne.io.RawArray:
    """
    perform the following pre-pre-processing steps on the raw data

    1. Notch filter at 50Hz
    2. High-pass filter at 1Hz [FIR filter]
    3. Re-refrence to average reference
    4. Amplitude tresholding
    """
    ############### filtering and re-refrencing ###############################

    # power line filtering: notch-filter at 50Hz, 100Hz and 150Hz
    raw.notch_filter(np.arange(50, 100, 150))

    # narrow-band filtering: filter the data
    raw_filtered = raw.copy().filter(
        l_freq=1.0, h_freq=None, method="fir", fir_design="firwin", skip_by_annotation="edge"
    )

    # re-ref to common-average
    raw_filtered.set_eeg_reference(ref_channels="average")

    # amplitude thresholding
    # i would suggest to exclude trials which have sample(s) exceeding +-150 micro volts

    return raw_filtered


def remove_eog(raw: mne.io.RawArray) -> mne.io.RawArray:
    """
    Remove EoG artifact based on template matching. choose a proxy channel for EoG artifact.
    Choose the channel that is nears to the eyes (usually frontal lobe)
    """

    ##### Removing EOG ###############################################

    # define the ica object to apply to the data
    ica = mne.preprocessing.ICA(n_components=None, method="picard", random_state=97)
    ica.fit(raw)  # raw is the raw data after re-refrencing and amplitude thresholoding
    ica.exclude = []

    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="Fz", threshold=0.5, measure="correlation")
    ica.exclude = eog_indices

    # reconstruct the EEG by apply the ica object exlcuding bad ICs
    eog_removed = ica.apply(raw)

    return eog_removed
