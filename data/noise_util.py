import numpy as np

def add_noise_to_data(data, snr):
    """
    Add Gaussian noise to data with a specific signal-to-noise ratio (SNR)
    
    Parameters:
    -----------
    data : numpy.ndarray
        The input data array with shape (n_samples, n_time_steps, n_channels)
    snr : float
        Signal-to-noise ratio in dB
    
    Returns:
    --------
    numpy.ndarray
        Data with added noise
    """
    # Make a copy of the data to avoid modifying the original
    noisy_data = data.copy()
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr / 10)
    
    # Add noise to each sample
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            # Get the signal
            signal = data[i, :, j]
            
            # Calculate signal power
            signal_power = np.mean(signal ** 2)
            
            # Calculate noise power based on SNR
            noise_power = signal_power / snr_linear
            
            # Generate Gaussian noise
            noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
            
            # Add noise to signal
            noisy_data[i, :, j] = signal + noise
    
    return noisy_data
