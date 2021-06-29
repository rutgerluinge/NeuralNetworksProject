from scipy.signal import butter, sosfilt
from scipy import fftpack, fft
import numpy as np
from ESN.ReferenceScripts.pyESN import ESN

# Get frequency components from timeseries
def frequencies(data, sample_frequency):
    spectrum = fft.fft(data)
    axis = fft.fftfreq(len(spectrum), sample_frequency)
    return spectrum, axis

# Low pass butterworth filter
def butter_lowpass(data, cutoff, sample_frequency, order=5):
    nyq = 0.5 * sample_frequency
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, data)

# High pass butterworth filter
def butter_highpass(data, cutoff, sample_frequency, order=5):
    nyq = 0.5 * sample_frequency
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sosfilt(sos, data)
    
# Band pass butterworth filter
def butter_bandpass(data, cutoff_low, cutoff_high, sample_frequency, order=5):
    nyq = 0.5 * sample_frequency
    sos = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band', analog=False, output='sos')
    return sosfilt(sos, data)

# Express entire dataset as fraction of first value in array
def normalize(data: np.ndarray):
    ref = data[0]
    normalized = np.empty((0))
    for i in range(data.size):
        normalized = np.append(normalized, data[i] / ref)
    return ref, normalized

def denormalize(ref, data: np.ndarray):
    denormalized = np.empty((0))
    for i in range(data.size):
        denormalized = np.append(denormalized, data[i] * ref)
    return denormalized

# Express dataset as as change relative to preceding value
# returned dataset is of size: input dataset - 1
def normalize_change(data: np.ndarray):
    # return first value in the sequence from which the subsequent values is expressed as relative change
    ref = data[0]
    normalized = np.empty((0))
    # replace zeros in the dataset by ones to prevent zero-division errors.
    data[data == 0] = 1
    for i in range(1, data.size):
        normalized = np.append(normalized, data[i] / data[i - 1])
    return ref, normalized

# Use the reference to convert relative change values to real values
def denormalize_change(ref, data: np.ndarray):
    denormalized = np.empty((1))
    denormalized[0] = ref * data[0]
    for i in range(1, data.size):
        denormalized = np.append(denormalized, denormalized[i - 1] * data[i])
    return denormalized

# Express dataset as change relative to preceding value
# returned dataset is of size: input dataset - 1
def difference(data):
    differenced = np.empty((0))
    ref = data[0]
    for i in range(1, data.size):
        differenced = np.append(differenced, data[i] - data[i - 1])
    return ref, differenced

# Use the reference to convert relative change values to real values
def dedifference(ref, data):
    dedifferenced = np.empty((1))
    dedifferenced[0] = ref + data[0]
    for i in range(1, data.size):
        dedifferenced = np.append(dedifferenced, dedifferenced[i - 1] + data[i])
    return dedifferenced

# Calculate Mean Squared Error (MSE)
def MSE(yhat, y):
    ph = np.subtract(yhat, y)
    ph = np.power(ph, 2)
    ph = np.mean(ph)
    ph = np.sqrt(ph)
    return ph

# Moving average smoothing
def moving_average(data: np.ndarray, window, downsampling = 1):
    """ 
    Smooth a timeseries by moving average
    Parameters:
    data: numpy ndarray
        array with only 1 dimension (n,), containing the timeseries.
    window: tuple
        the window over which the mean value should be calculated, 
        with the first element as the number of samples before the current position,
        and the second element as the number of samples after the current position
        Ex: (2,2), (0,4)
    """
    data_length = len(data)
    new_data = np.empty((0))
    for i in range(downsampling - 1, data_length, downsampling):
        shift = -min(max(0, (i + window[1] + 1) - data_length), i - window[0])
        start_index = i - window[0] + shift
        stop_index = i + window[1] + 1 + shift
        mean = np.mean(data[start_index:stop_index])
        new_data = np.append(new_data, mean)
    return new_data

# Model agency
def make_modell_2(reservoir_size, sparsity, spectral_radius, noise, random_seed = 20):
    das_modell = ESN(
        n_inputs = 1,
        n_outputs = 1,
        n_reservoir = reservoir_size,
        sparsity = sparsity,
        random_state = random_seed,
        spectral_radius = spectral_radius,
        noise = noise,
        silent = True
    )
    return das_modell

def make_modell(hyperparameters: dict):
    das_modell = ESN(
        n_inputs = 1,
        n_outputs = 1,
        n_reservoir = hyperparameters['resevoir_size'],
        sparsity = hyperparameters['sparsity'],
        random_state = hyperparameters['rand_seed'],
        spectral_radius = hyperparameters['spectral_radius'],
        noise = hyperparameters['noise'],
        silent = True
    )
    return das_modell