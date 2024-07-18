import numpy as np
from scipy import signal


class FilterBankTransformer:
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, banks, fs, filter_allowance=2, axis=1, filter_type="filter"):
        self.banks = banks
        self.fs = fs
        self.filter_allowance = filter_allowance
        self.axis = axis
        self.filter_type = filter_type

    @staticmethod
    def bandpass_filter(data, band_filt_cut_f, fs, filter_allowance=2, axis=1, filter_type="filter"):
        """
         Filter a signal using cheby2 iir filtering.

        Args:
            data: 2d/ 3d np array
                trial x channels x time
            bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
                if any value is specified as None then only one sided filtering will be performed
            fs: sampling frequency
            filtAllowance: transition bandwidth in hertz
            filtType: string, available options are 'filtfilt' and 'filter'

        Returns:
            dataOut: 2d/ 3d np array after filtering
                Data after applying bandpass filter.
        """
        a_stop = 30  # stopband attenuation
        a_pass = 3  # passband attenuation
        n_freq = fs / 2  # Nyquist frequency

        if not band_filt_cut_f[0] and (not band_filt_cut_f[1] or (band_filt_cut_f[1] >= fs / 2.0)):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif band_filt_cut_f[0] == 0 or band_filt_cut_f[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            f_pass = band_filt_cut_f[1] / n_freq
            f_stop = (band_filt_cut_f[1] + filter_allowance) / n_freq
            # find the order
            [N, ws] = signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
            b, a = signal.cheby2(N, a_stop, f_stop, "lowpass")

        elif (band_filt_cut_f[1] is None) or (band_filt_cut_f[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            f_pass = band_filt_cut_f[0] / n_freq
            f_stop = (band_filt_cut_f[0] - filter_allowance) / n_freq
            # find the order
            [N, ws] = signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
            b, a = signal.cheby2(N, a_stop, f_stop, "highpass")
        else:
            # band-pass filter
            # print("Using bandpass filter")
            f_pass = (np.array(band_filt_cut_f) / n_freq).tolist()
            f_stop = [
                (band_filt_cut_f[0] - filter_allowance) / n_freq,
                (band_filt_cut_f[1] + filter_allowance) / n_freq,
            ]
            # find the order
            [N, ws] = signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
            b, a = signal.cheby2(N, a_stop, f_stop, "bandpass")

        if filter_type == "filtfilt":
            return signal.filtfilt(b, a, data, axis=axis)
        else:
            return signal.lfilter(b, a, data, axis=axis)

    def __call__(self, data):
        # initialize output
        filter_banked_signals = np.zeros([*data.shape, len(self.banks)])

        # repetitively filter the data.
        for i, filter_band in enumerate(self.banks):
            filter_banked_signals[:, :, i] = self.bandpass_filter(
                data, filter_band, self.fs, self.filter_allowance, self.axis, self.filter_type
            )

        # remove any redundant 3rd dimension
        if len(self.banks) <= 1:
            filter_banked_signals = np.squeeze(filter_banked_signals, axis=2)

        return filter_banked_signals
