import numpy as np
import matplotlib.pyplot as plt 
from math import pi

def dft(signal : np.array) -> np.array:
    """Returns the Discrete Fourier Transform of a signal

    Args:
        signal (np.array): Input signal

    Returns:
        np.array: Input Signal in the Frequency Domain
    """

    #1. Generate the DFT matrix by completing the <create_dft_matrix> function.
    length_signal = len(signal)
    dft_matrix = create_dft_matrix(length=length_signal)

    #2. Apply the DFT matrix to the signal by completing the <apply_dft_matrix> function.
    #   Store the frequency domain representation of the signal in variable <F_Signal>
    F_signal = apply_dft_matrix(dft_matrix=dft_matrix, signal=signal)

    return F_signal

def create_dft_matrix(length : int) -> np.array:
    """Returns a DFT matrix to enable calculation of the DFT.

    Args:
        length (int): length of the input signal or N of the NxN DFT Matrix.

    Returns:
        np.array: NxN DFT Matrix
    """
    dft_mat = np.zeros(shape=(length,length), dtype=np.cdouble)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: The complex number e^(-4j) can be represented in numpy by <np.exp(-4j)>
    
    # dft_mat = np.fft.fft(np.eye(length)) should have the same plot
    
    N = length
    phi_ = 2*pi/N
    phi_j = complex(0, phi_)
    W_N = np.exp(-phi_j)
    for m in range(length):
        for n in range(length):
            dft_mat[m][n] = pow(W_N,m*n)

    #***************************** Please add your code implementation above this line *****************************

    return dft_mat

#give the generated dft_mat matrix as the input
def apply_dft_matrix(dft_matrix : np.array, signal : np.array):
    """_summary_

    Args:
        dft_matrix (np.array): The DFT Matrix
        signal (np.array): The time-domain signal

    Returns:
        F_signal (np.array): The frequency-domain signal
    """
    signal_length = signal.shape[0]
    F_signal = np.zeros(shape=signal_length)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: search the numpy library documentation for the <np.matmul> operation.
    # matmul broadcasts the last 2 rows of the matrix
    F_signal = np.matmul(signal, dft_matrix)
    
    #***************************** Please add your code implementation above this line *****************************

    return F_signal #returns the DFT of the input signal

def plot_dft_magnitude_angle(frequency_axis : np.array, f_signal : np.array, fs : int=1, format : str=None):
    """_summary_

    Args:
        frequency_axis (np.array).
        f_signal (np.array): input signal in the frequency domain
        fs (int, optional): Sampling Frequency. Defaults to 1.
        format (string, optional): Plotting Format. Defaults to None.
    """
    
    N = len(f_signal)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    #***************************** Please add your code implementation under this line *****************************
    frequency_axis_normalized = []
    f_signal_normalized = []
    
    if(format == None):
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(frequency_axis, np.angle(f_signal))
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))

    if(format == "ZeroPhase"):
        #Hint: the numpy library documentation for the <np.where> operation might be useful.
        f_signal_normalized = np.where(np.abs(f_signal) < 0.1, 0.00, f_signal)
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal_normalized))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(frequency_axis, np.angle(f_signal_normalized))
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
        f_signal_normalized = [] #Clear Buffer
        
        
    if(format == "Normalized"):
        f_signal_normalized = np.where(np.abs(f_signal) < 0.1, 0.00, f_signal)
        
        for i in range(len(frequency_axis)):
            frequency_axis_normalized.append(frequency_axis[i]/len(frequency_axis))
               
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis_normalized, np.abs(f_signal_normalized))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(frequency_axis_normalized, np.angle(f_signal_normalized))
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
        f_signal_normalized = []
        frequency_axis_normalized = []
        

    if(format == "Centered_Normalized"):
        if len(f_signal)%2 == 0:
            buffer = f_signal[int(len(f_signal)/2):]
            buffer = np.where(np.abs(buffer) < 0.1, 0.00, buffer)
            f_signal_normalized = np.where(np.abs(f_signal) < 0.1, 0.00, f_signal)
           
            for i in range(int(len(f_signal_normalized)/2)):
                f_signal_normalized[i+int(len(f_signal_normalized)/2)] = f_signal_normalized[i]
                f_signal_normalized[i] = buffer[i] 
        
            for i in range(len(frequency_axis)):
                frequency_axis_normalized.append(frequency_axis[i]/len(frequency_axis)-0.5)

            ax1.set_ylabel("Magnitude")
            ax1.stem(frequency_axis_normalized, np.abs(f_signal_normalized))
            ax2.set_ylabel("Phase (radians)")
            ax2.stem(frequency_axis_normalized, np.angle(f_signal_normalized))
            ax2.set_xlabel("Frequency Bins")
            ax2.set_ylim((-np.pi, np.pi))
            f_signal_normalized = []
            frequency_axis_normalized = []

    if(format == "Centered_Original_Scale"):
        if len(f_signal)%2 == 0:
            buffer = f_signal[int(len(f_signal)/2):]
            buffer = np.where(np.abs(buffer) < 0.1, 0.00, buffer)
            f_signal_normalized = np.where(np.abs(f_signal) < 0.1, 0.00, f_signal)
           
            for i in range(int(len(f_signal_normalized)/2)):
                f_signal_normalized[i+int(len(f_signal_normalized)/2)] = f_signal_normalized[i]
                f_signal_normalized[i] = buffer[i] 
        
            for i in range(len(frequency_axis)):
                frequency_axis_normalized.append(fs*(frequency_axis[i]/len(frequency_axis)-0.5))

            ax1.set_ylabel("Magnitude")
            ax1.stem(frequency_axis_normalized, np.abs(f_signal_normalized))
            ax2.set_ylabel("Phase (radians)")
            ax2.stem(frequency_axis_normalized, np.angle(f_signal_normalized))
            ax2.set_xlabel("Frequency Bins")
            ax2.set_ylim((-np.pi, np.pi))
            f_signal_normalized = []
            frequency_axis_normalized = []

    #***************************** Please add your code implementation above this line *****************************

def idft(signal : np.array) -> np.array:
    """Returns the Inverse Discrete Fourier Transform of a frequency domain signal. 

    Args:
        signal (np.array): Input frequency signal

    Returns:
        np.array: Transforms input signal to time-domain.
    """
    length_signal = len(signal)
    time_domain_signal = np.zeros(length_signal)

    #***************************** Please add your code implementation under this line *****************************
    # Try to only use the <create_dft_matrix> and <apply_dft_matrix> operations that you have already implemented and some numpy operations.
    # Hint: look up the <np.conjugate>
    idft_mat = np.zeros(shape=(length_signal,length_signal), dtype=np.cdouble)
    
    N = length_signal
    phi_ = 2*pi/N
    phi_j = complex(0, phi_)
    W_N = np.exp(-phi_j)
    W_N_inv = np.conjugate(W_N)
    for m in range(length_signal):
        for n in range(length_signal):
            idft_mat[m][n] = pow(W_N_inv,m*n)
    
    time_domain_signal = (1/N)*np.matmul(signal, idft_mat)
    #***************************** Please add your code implementation above this line *****************************

    return time_domain_signal

def convolve_signals(x_signal : np.array, y_signal : np.array) -> np.array:
    """Convolve the X and Y signal using the DFT.

    Args:
        x_signal (np.array)
        y_signal (np.array)

    Returns:
        np.array: z_signal = x_signal * y_signal
    """
    z_signal = np.zeros(len(x_signal))
    
    #***************************** Please add your code implementation under this line *****************************
    #Make sure you do the time-domain convolution in the frequency domain. 
    X_w = dft(x_signal)
    Y_w = dft(y_signal)
    Z_w = np.multiply(X_w, Y_w)
    z_signal = idft(Z_w)
    #***************************** Please add your code implementation above this line *****************************

    return z_signal

def zero_pad_signal(x_signal : np.array, new_length : int) -> np.array:
    """Append zeros to the end of the input signal until it reaches the desired [new_length].

    Args:
        x_signal (np.array): input signal.
        new_length (int): new length of signal such that we add new_length minus original_length number of zeros to the end of the signal

    Returns:
        np.array: zero-padded signal
    """
    zero_padded_signal = np.zeros(new_length)

    #***************************** Please add your code implementation under this line *****************************
    for i in range(len(x_signal)):
        zero_padded_signal[i] = x_signal[i]
    #***************************** Please add your code implementation above this line *****************************

    return zero_padded_signal
