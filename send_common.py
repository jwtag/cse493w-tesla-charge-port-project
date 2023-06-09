import SoapySDR
import numpy as np
from SoapySDR import *  # import SoapySDR constants

#
# Contains code shared across the send_* files.
#

def send_samples(samples_file_name, fsps, radio_freq):
    """
    Loads the samples stored at the passed filename + sends them over the radio as specified.
    """

    # attempt to load the radio recording from disk.
    samples = np.load(samples_file_name)
    if not samples.any():
        raise Exception("failed to load samples from disk!")

    print("loaded samples from disk successfully")

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
    sdr.setSampleRate(SOAPY_SDR_TX, 0, fsps)
    sdr.setFrequency(SOAPY_SDR_TX, 0, radio_freq)   
    sdr.setGain(SOAPY_SDR_TX, 0, 47)
    txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])

    # chop-up the samples into parts that are smaller than the antenna's mtu
    mtu = sdr.getStreamMTU(txStream)
    num_per_buf = (2 * len(samples)) / mtu
    samples_split = np.split(samples, num_per_buf)  # split it into chunks that are about (0.5 * mtu) in size.

    # send out saved Tesla BLE key samples
    sdr.activateStream(txStream) # start stream to send out recording.
    for split in samples_split:  # send out each split one-by-one.
        num_wr = sdr.writeStream(txStream, [split], len(split))
    sdr.deactivateStream(txStream)
    sdr.closeStream(txStream)

    print("samples successfully sent!")