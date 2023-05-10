"""
Records a nearby Tesla charger radio transmission + saves it to a file for replay later.
"""
import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import *  # import SoapySDR constants
from copy import copy

RECORDING_LENGTH = 5  # the length of the recording we're taking in seconds.
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


    def process_samples(samples):
        """
        Processes up the provided samples to be clearer using bandpass filtering, squelching, and downsampling.

        NOTE:  This was derived from my HW4 code.

        :param samples: the samples being processed.
        :return: a cleaned-up version of the samples.
        """
        print("processing samples")

        print("filtering with bandpass mask")

        # get a bandpass mask.
        fcutoff = 200000 # Cutoff frequency of filter 100kHz
        bpm = _get_taperedbandpassmask(fcutoff, 200000)

        # fftshift samples to get a spectrum of data.
        spectrum = np.fft.fftshift(np.fft.fft(samples))

        # filter spectrum w/ bandpass mask.
        filteredspectrum = spectrum * bpm

        # Convert masked spectrum back to time domain to get filtered signal 
        filteredsignal = np.fft.ifft(np.fft.fftshift(filteredspectrum))

        # get theta from filtered signal
        theta = np.arctan2(filteredsignal.imag,filteredsignal.real)

        # setup various arrays used for squelching signal.
        abssignal = np.abs(filteredsignal)
        meanabssignal = np.mean(abssignal)
        thetasquelched = copy(theta)
        for i in range(NUM_SAMPLES):
            if (abssignal[i]<(meanabssignal/3.0)):
                thetasquelched[i] = 0.0
        derivthetap0 = np.convolve([1,-1],theta,'same')
        derivthetapp = np.convolve([1,-1],(theta+np.pi) % (2*np.pi),'same')

        print("bandpass filter complete.  squelching signal...")

        # do squelching...
        derivthetap0[100:110]-derivthetapp[100:110]
        derivthetap0 = np.convolve([1,-1],thetasquelched,'same')
        derivthetapp = np.convolve([1,-1],(thetasquelched+np.pi) % (2*np.pi),'same')
        # The 0, +pi comparison method
        # deriv (theta plus pi)
        derivtheta = np.zeros(len(derivthetap0))
        for i in range(len(derivthetap0)):
            if (abs(derivthetap0[i])<abs(derivthetapp[i])):
                derivtheta[i] = derivthetap0[i] 
            else:
                derivtheta[i] = derivthetapp[i] 
        cdtheta = copy(derivtheta)
        spikethresh = 2
        cdtheta = copy(derivtheta) # Cleaned derivative of theta
        for i in range(1,len(derivtheta)-1):
            if (abs(derivtheta[i])>spikethresh):
                cdtheta[i] = (derivtheta[i-1]+derivtheta[i+1])/2.0

        # now that squelching is complete, let's downsample
        dsf = round(FSPS / SAMPLING_FREQ) # round(1048576/48000)=22
        dscdtheta = cdtheta[::dsf] # downsample by 22 (or whatever dsf is)
        dscdtheta2 = copy(dscdtheta)
        for i in range(len(dscdtheta2)):
            dscdtheta2[i] = np.sum(cdtheta[i*dsf:(i+1)*dsf])/dsf
        dscdtheta = copy(dscdtheta2)

        print("squelching complete, returning processed samples.")

        # plot the collected samples.
        _plot_samples(dscdtheta)

        return dscdtheta


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

    def _get_taperedbandpassmask(fcutoff,xwidth):
        """
        Creates + returns a bandpass mask which meets the provided param specifications.

        Taken from CSE493W Jupyter notebook.
        """
        fcutoff_n = fcutoff / FSPS # fcutoff, normalized
        xwidth_n = xwidth / FSPS # transition width, normalized
        
        pbfw = round(2*fcutoff_n * NUM_SAMPLES)
        xbw = round(xwidth_n * NUM_SAMPLES)
        sbw = int((NUM_SAMPLES-pbfw-xbw-xbw)/2)
        res = np.concatenate((np.zeros(sbw), #
                            np.arange(0.0,1.0,1.0/xbw), #
                            np.ones(pbfw), #
                            np.arange(1.0,0.0,-1.0/xbw), #
                            np.zeros(sbw)))
        return(res)

    def _verify_saved_samples(samples):
        """
        Verifies that the passed samples were successfully saved in SAMPLES_OUTPUT_FILE_NAME.
        """
        loaded_samples = np.load(SAMPLES_OUTPUT_FILE_NAME)
        assert np.array_equal(samples, loaded_samples)
        print("verified that data was saved correctly to " + SAMPLES_OUTPUT_FILE_NAME)

    # record the samples.
    samples = record_samples()

    # process the samples (we want to clean them up before storage).
    cleaned_samples = process_samples(samples)

    # save the samples.
    save_samples(cleaned_samples)