"""
Records a nearby Tesla charger radio transmission + saves it to a file for replay later.
"""
import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import *  # import SoapySDR constants
from copy import copy

RECORDING_LENGTH = 10  # the length of the recording we're taking.
                         # This should be long enough to record the entire transmission.
RADIO_FREQ = 315e6  # freq which we're recording upon (in Hz).  Tesla charger radio freq is 315mHz
FSPS = 2 * 256 * 256 * 16  # record at about 2Msps
SAMPLES_OUTPUT_FILE_NAME = "saved_samples.npy"  # .npy == the official Numpy binary output file extension.
NUM_SAMPLES = round(FSPS * RECORDING_LENGTH)
SAMPLING_FREQ = 48000

if __name__ == "__main__":
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

        # setup hackrf sdr object
        args = dict(driver="hackrf")
        sdr = SoapySDR.Device(args)

        # config sdr object
        sdr.setSampleRate(SOAPY_SDR_RX, 0, FSPS)
        sdr.setFrequency(SOAPY_SDR_RX, 0, RADIO_FREQ)   

        # create numpy array to store samples.
        samples = np.array([0] * NUM_SAMPLES, np.complex64)

        # collect samples
        rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        sdr.activateStream(rxStream) # start streaming
        sdr.readStream(rxStream, [samples], len(samples))
        sdr.deactivateStream(rxStream)
        sdr.closeStream(rxStream)

        print("samples successfully recorded!")

        # plot the collected samples.
        _plot_samples(samples)

        # return the samples.
        return samples


    def save_samples(samples):
        """
        Saves the samples to the file whose name is specified by SAMPLE_OUTPUT_FILE_NAME.
        :param samples: the samples being saved.
        :return: None.
        """

        # write to the file.
        np.save(SAMPLES_OUTPUT_FILE_NAME, samples)

        print("successfully saved samples to " + SAMPLES_OUTPUT_FILE_NAME)

        # verify that the data was saved as expected.
        _verify_saved_samples(samples)

    def _plot_samples(samples):
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

    def _verify_saved_samples(samples):
        """
        Verifies that the passed samples were successfully saved in SAMPLES_OUTPUT_FILE_NAME.
        """
        loaded_samples = np.load(SAMPLES_OUTPUT_FILE_NAME)
        assert np.array_equal(samples, loaded_samples)
        print("verified that data was saved correctly to " + SAMPLES_OUTPUT_FILE_NAME)

    # record the samples.
    samples = record_samples()

    # save the samples.     
    save_samples(samples)