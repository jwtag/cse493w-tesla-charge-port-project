import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import *  # import SoapySDR constants
from copy import copy

#
# Contains code shared across the record_* files.
#

def record_samples(radio_freq, fsps, num_samples):
    """
    Records samples on the frequency specified by RADIO_FREQ + returns the unmodified recorded sample data.

    NOTE:  Much of the below code is derived from Jupyter notebooks used in class.

    :return: the unmodified recorded sample data as a Numpy array.
    """
    print("now recording samples at " + str(radio_freq) + " Hz.  Please trigger the signal you wish to record.")

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
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fsps)
    sdr.setFrequency(SOAPY_SDR_RX, 0, radio_freq)   

    # create numpy array to store samples.
    samples = np.array([0] * num_samples, np.complex64)

    # collect samples
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream) # start streaming
    sdr.readStream(rxStream, [samples], len(samples))
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

    print("samples successfully recorded!")

    # plot the collected samples.
    _plot_samples(samples, fsps, radio_freq)

    # return the samples.
    return samples

def process_samples(samples, fsps, num_samples, sampling_freq, radio_freq):
        """
        Processes up the provided samples to be clearer using bandpass filtering, squelching, and downsampling.
        NOTE:  This was derived from my HW4 code.
        :param samples: the samples being processed.
        :return: a cleaned-up version of the samples.
        """
        print("processing samples")

        print("filtering with bandpass mask")

        # get a bandpass mask.
        fcutoff = 100000 # Cutoff frequency of filter 100kHz
        bpm = _get_taperedbandpassmask(fcutoff, 200000, fsps, num_samples)

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
        for i in range(num_samples):
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
        dsf = round(fsps / sampling_freq) # round(1048576/48000)=22
        dscdtheta = cdtheta[::dsf] # downsample by 22 (or whatever dsf is)
        dscdtheta2 = copy(dscdtheta)
        for i in range(len(dscdtheta2)):
            dscdtheta2[i] = np.sum(cdtheta[i*dsf:(i+1)*dsf])/dsf
        dscdtheta = copy(dscdtheta2)

        print("squelching complete, returning processed samples.")

        # plot the collected samples.
        _plot_samples(dscdtheta, fsps, radio_freq)

        return dscdtheta

def _plot_samples(samples, fsps, radio_freq):
    """
    Plots out the samples on a plot.

    Useful for debugging.

    :param samples: a Numpy array of the the samples being plotted.
    :param N:  the number of samples collected
    :return: None.
    """
    nyquist = fsps / 2.0
    plt.figure()
    spectrum = np.fft.fftshift(np.fft.fft(samples))
    maxval = np.amax(np.abs(spectrum))
    maxindi = np.argmax(np.abs(spectrum))
    freqs = np.linspace(radio_freq - nyquist, radio_freq + nyquist, len(spectrum))
    plt.plot(freqs / 1e6, np.abs(spectrum))
    print("For band from ", (radio_freq - nyquist) / 1e6, " to ", (radio_freq + nyquist) / 1e6, "MHz,")
    print(" max power is", (maxval ** 2), ", at ", freqs[maxindi] / 1e6, "MHz (index ", round(maxindi), ")")
    plt.show()

def _get_taperedbandpassmask(fcutoff, xwidth, fsps, num_samples):
    """
    Creates + returns a bandpass mask which meets the provided param specifications.
    Taken from CSE493W Jupyter notebook.
    """
    fcutoff_n = fcutoff / fsps # fcutoff, normalized
    xwidth_n = xwidth / fsps # transition width, normalized

    pbfw = round(2*fcutoff_n * num_samples)
    xbw = round(xwidth_n * num_samples)
    sbw = int((num_samples-pbfw-xbw-xbw)/2)
    res = np.concatenate((np.zeros(sbw), #
                        np.arange(0.0,1.0,1.0/xbw), #
                        np.ones(pbfw), #
                        np.arange(1.0,0.0,-1.0/xbw), #
                        np.zeros(sbw)))
    return(res)

def save_samples(samples, output_file_name):
    """
    Saves the samples to the file whose name is specified by SAMPLE_OUTPUT_FILE_NAME.
    :param samples: the samples being saved.
    :return: None.
    """

    # write to the file.
    np.save(output_file_name, samples)

    print("successfully saved samples to " + output_file_name)

    # verify that the data was saved as expected.
    loaded_samples = np.load(output_file_name)
    assert np.array_equal(samples, loaded_samples)
    print("verified that data was saved correctly to " + output_file_name)
