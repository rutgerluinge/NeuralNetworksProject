from scipy.signal import butter, sosfilt
from scipy import fftpack, fft
import numpy as np
from ESN.ReferenceScripts.pyESN import ESN

# Get frequency components from timeseries
def frequencies(data, sample_frequency):
    """
    Calculate the frequency components of the input signal by Fast Fourier Transform
    Parameters:
    ----------
    data : ndarray
        Input timeseries
    
    sample_frequency : float
        frequency of the samples in the timeseries.
    
    Returns:
    ----------
    spectrum : ndarray
        The frequency components of the input timeseries.
    
    axis : 
        The axis of the frequency components, used for plotting.
    """
    spectrum = fft.fft(data)
    axis = fft.fftfreq(len(spectrum), sample_frequency)
    return spectrum, axis

# Low pass butterworth filter
def butter_lowpass(data, cutoff, sample_frequency, order=5):
    """ 
    Apply a low-pass butterworth filter to the timeseries.
    
    Parameters:
    ----------
    data : ndarray
        Array of normalized values
    
    cutoff : float
        Frequency along which to filter the data. All frequency components above this value will
        be filtered from the input data.
    
    sample_frequency : float
        frequency of the samples in the timeseries.
    
    order : int
        Order of the butterworth filter
    
    Returns:
    ----------
    filtered : ndarray
        The filtered timeseries
    """
    nyq = 0.5 * sample_frequency
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, data)

# High pass butterworth filter
def butter_highpass(data, cutoff, sample_frequency, order=5):
    """ 
    Apply a high-pass butterworth filter to the timeseries.
    
    Parameters:
    ----------
    data : ndarray
        Array of normalized values
    
    cutoff : float
        Frequency along which to filter the data. All frequency components below this value will
        be filtered from the input data.
    
    sample_frequency : float
        frequency of the samples in the timeseries.
    
    order : int
        Order of the butterworth filter
    
    Returns:
    ----------
    filtered : ndarray
        The filtered timeseries
    """
    nyq = 0.5 * sample_frequency
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sosfilt(sos, data)
    
# Band pass butterworth filter
def butter_bandpass(data, cutoff_low, cutoff_high, sample_frequency, order=5):
    """ 
    Apply a low-pass butterworth filter to the timeseries.
    
    Parameters:
    ----------
    data : ndarray
        Array of normalized values
    
    cutoff_low : float
        Frequency along which to filter the data. All frequency components below this value will
        be filtered from the input data.
    
    cutoff_high : float
        Frequency along which to filter the data. All frequency components above this value will
        be filtered from the input data.
    
    sample_frequency : float
        frequency of the samples in the timeseries.
    
    order : int
        Order of the butterworth filter
    
    Returns:
    ----------
    filtered : ndarray
        The filtered timeseries
    """
    nyq = 0.5 * sample_frequency
    sos = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band', analog=False, output='sos')
    return sosfilt(sos, data)

# Express entire dataset as fraction of first value in array
def normalize(data: np.ndarray):
    """
    Express timeseries as fraction of first value in the timeseries.
    The returned dataset is of size: input dataset - 1.
    
    Parameters:
    ----------
    data : ndarray
        data to convert.
    
    Returns:
    ----------
    ref : any
        The first value in the input timeseries, which is required to 
        reconstruct the input timeseries from the normalized timeseries.
    
    normalized : ndarray
        The normalized timeseries
    """
    ref = data[0]
    normalized = np.empty((0))
    for i in range(data.size):
        normalized = np.append(normalized, data[i] / ref)
    return ref, normalized

# Convert a series of normalized values back to 'normal' timeseries.
def denormalize(ref, data: np.ndarray):
    """ 
    Convert a series of normalized values back to 'normal' timeseries.
    
    Parameters:
    ----------
    ref : any
        Starting value of the new 'normal' timeseries. Is used to calculate 
        the subsequent values: new[0] = ref * norm[0], new[1] = ref * norm[1]
    
    data : ndarray
        Array of normalized values
    
    Returns:
    ----------
    denormalized : ndarray
        The denormalized timeseries
    """
    denormalized = np.empty((0))
    for i in range(data.size):
        denormalized = np.append(denormalized, data[i] * ref)
    return denormalized

# Express dataset as a fraction of the preceding value.
# The returned dataset is of size: input dataset - 1
def normalize_change(data: np.ndarray):
    """
    Express dataset as a fraction of the preceding value
    returned dataset is of size: input dataset - 1
    
    Parameters:
    ----------
    data : ndarray
        data to convert.
    
    Returns:
    ----------
    ref : any
        The first value in the input timeseries, which is required to 
        reconstruct the input timeseries from the normalized timeseries.
    
    normalized : ndarray
        The normalized timeseries
    """
    # return first value in the sequence from which the subsequent values is expressed as relative change
    ref = data[0]
    normalized = np.empty((0))
    # replace zeros in the dataset by ones to prevent zero-division errors.
    # This modification leads to a neglegible error on the range used here (-3000, 10000)
    data[data == 0] = 1
    for i in range(1, data.size):
        normalized = np.append(normalized, data[i] / data[i - 1])
    return ref, normalized

# Use the reference to convert relative change values to real values
def denormalize_change(ref, data: np.ndarray):
    """ 
    Convert a series of normalized values back to 'normal' timeseries.
    
    Parameters:
    ----------
    ref : any
        Starting value of the new 'normal' timeseries. Is used to calculate 
        the subsequent values: new[0] = ref * norm[0], new[1] = new[0] * norm[1]
    
    data : ndarray
        Array of normalized values
    
    Returns:
    ----------
    denormalized : ndarray
        The denormalized timeseries
    """
    denormalized = np.empty((1))
    denormalized[0] = ref * data[0]
    for i in range(1, data.size):
        denormalized = np.append(denormalized, denormalized[i - 1] * data[i])
    return denormalized

# Express dataset as change relative to preceding value
# returned dataset is of size: input dataset - 1
def difference(data):
    """
    Express the input timeseries as a series of differences, aka each 
    value in the differenced array is the change in value compared to the 
    preceding value.
    
    Parameters:
    ----------
    data : ndarray
        data to convert.
    
    Returns:
    ----------
    ref : any
        The first value in the input timeseries, which is required to 
        reconstruct the input timeseries from the differenced timeseries.
    
    differenced : ndarray
        The differenced timeseries
    """
    differenced = np.empty((0))
    ref = data[0]
    for i in range(1, data.size):
        differenced = np.append(differenced, data[i] - data[i - 1])
    return ref, differenced

# Use the reference to convert relative change values to real values
def dedifference(ref, data):
    """ 
    Convert a series of differenced values back to 'normal' timeseries.
    
    Parameters:
    ----------
    ref : any
        Value preceding the new 'normal' timeseries. Is used to calculate 
        the subsequent values: new[0] = ref + diff[0], new[1] = new[0] + diff[1]
    
    data : ndarray
        Array of differenced values
    
    Returns:
    ----------
    dedifferenced : ndarray
        The dedifferenced timeseries
    """
    dedifferenced = np.empty((1))
    dedifferenced[0] = ref + data[0]
    for i in range(1, data.size):
        dedifferenced = np.append(dedifferenced, dedifferenced[i - 1] + data[i])
    return dedifferenced

# Calculate Mean Squared Error (MSE)
def MSE(yhat, y):
    """
    Calculate the Mean Squared Error from the 2 input timeseries.
    
    Parameters:
    ----------
    yhat : ndarray
        the artificially produced timeseries which are to be compared to the original timeseries.
    
    y : ndarray
        The original timeseries to compare `yhat` with.
    
    Returns:
    ----------
    MSE : float
        The Mean Squared Error
    """
    ph = np.subtract(yhat, y)
    ph = np.power(ph, 2)
    ph = np.mean(ph)
    ph = np.sqrt(ph)
    return ph

# Moving average smoothing
def moving_average(data: np.ndarray, window, downsampling = 1):
    """ 
    Smooth a timeseries with a moving average.
    
    Parameters:
    ----------
    data : numpy ndarray
        Array with only 1 dimension (n,), containing the timeseries.
    
    window : tuple
        The window over which the mean value should be calculated, 
        with the first element as the number of samples before the current position,
        and the second element as the number of samples after the current position
        Ex: (2,2), (0,4).
    
    downsampling : int
        The factor by which to scale down the timeseries. 
        Ex: downsampling = 2 divides the total number of elements in the timeseries by 2
    
    Returns:
    ----------
    smoothed : ndarray
        The smoothed timeseries
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
def make_modell_2(reservoir_size, sparsity, spectral_radius, noise = 0.001, random_seed = 20):
    """
    Generate a echo state network with the supplied parameters
    
    Returns:
    ----------
    model : pyESN.ESN
        An echo state network model object
    """
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

# Model agency
def make_modell(hyperparameters: dict):
    """
    Generate a echo state network with a hyperparameter dictionary
    
    Returns:
    ----------
    model : pyESN.ESN
        An echo state network model object
    """
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