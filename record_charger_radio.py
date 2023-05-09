"""
Records a nearby Tesla charger radio transmission + saves it to a file for replay later.
"""
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

RECORDING_LENGTH = 1  # the length of the recording we're taking in seconds.
                       # This should be long enough to record the entire transmission.
RADIO_FREQ = 94.9e6  # freq which we're recording upon (in Hz)
FSPS = 2 * 256 * 256 * 16  # record at about 2Msps
SAMPLES_OUTPUT_FILE_NAME = "saved_samples.teslasignal"


def record_samples():
    """
    Records samples on the frequency specified by RADIO_FREQ + returns the unmodified recorded sample data.

    NOTE:  Much of the below code is derived from Jupyter notebooks used in class.

    :return: the unmodified recorded sample data as a Numpy array.
    """
    print("now recording samples at " + str(RADIO_FREQ) + " Hz.  Please trigger the signal you wish to record.")

    # Close previous instances of the sdr object
    try:
        sdr.close()
        print("Closed old SDR")
    except NameError:
        print("No SDR instance found")

    # open up the sdr object
    sdr = RtlSdr()

    # config sdr object
    sdr.sample_rate = FSPS
    sdr.center_freq = RADIO_FREQ

    # calculate number of samples we wish to collect.
    N = round(FSPS * RECORDING_LENGTH)

    # collect the samples
    samples = sdr.read_samples(N)  # Collect N samples.

    print("samples successfully recorded!")

    # plot the collected samples.
    # TODO:  remove this in final product.
    _plot_samples(samples, N)

    # return the samples.
    return samples


def process_samples(samples):
    """
    Processes up the provided samples to be clearer.

    :param samples: the samples being processed.
    :return: a cleaned-up version of the samples.
    """
    # TODO:  implement this
    print("processing samples")
    return samples


def save_samples(samples):
    """
    Saves the samples to the file whose name is specified by SAMPLE_OUTPUT_FILE_NAME.
    :param samples: the samples being saved.
    :return: None.
    """
    # TODO:  implement this


def _plot_samples(samples, N):
    """
    Plots out the samples on a plot.

    Useful for debugging.

    :param samples: a Numpy array of the the samples being plotted.
    :param N:  the number of samples collected
    :return: None.
    """
    nyquist = FSPS / 2.0
    plt.figure()
    spectrum = np.fft.fftshift(np.fft.fft(samples))
    maxval = np.amax(np.abs(spectrum))
    maxindi = np.argmax(np.abs(spectrum))
    freqs = np.linspace(RADIO_FREQ - nyquist, RADIO_FREQ + nyquist, len(spectrum))
    plt.plot(freqs / 1e6, np.abs(spectrum))
    print("For band from ", (RADIO_FREQ - nyquist) / 1e6, " to ", (RADIO_FREQ + nyquist) / 1e6, "MHz,")
    print(" max power is", (maxval ** 2), ", at ", freqs[maxindi] / 1e6, "MHz (index ", round(maxindi), ")")
    plt.show()


# record the samples.
samples = record_samples()

# process the samples (we want to clean them up before storage).
cleaned_samples = process_samples(samples)

# save the samples.
save_samples(cleaned_samples)