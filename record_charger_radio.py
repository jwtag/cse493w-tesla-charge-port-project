"""
Records a nearby Tesla charger radio transmission + saves it to a file for replay later.
"""
from record_common import process_samples, record_samples, save_samples

RECORDING_LENGTH = 10  # the length of the recording we're taking.
                         # This should be long enough to record the entire transmission.
RADIO_FREQ = 315e6  # freq which we're recording upon (in Hz).  Tesla charger radio freq is 315mHz
FSPS = 2 * 256 * 256 * 16  # record at about 2Msps
SAMPLES_OUTPUT_FILE_NAME = "saved_samples.npy"  # .npy == the official Numpy binary output file extension.
NUM_SAMPLES = round(FSPS * RECORDING_LENGTH)
SAMPLING_FREQ = 48000

if __name__ == "__main__":
    # record the samples.
    samples = record_samples(RADIO_FREQ, FSPS, NUM_SAMPLES)

    # process samples
    # commented out this line since processing is currently not working...
    # processed_samples = process_samples(samples, FSPS, NUM_SAMPLES, SAMPLING_FREQ, RADIO_FREQ)

    # save the samples.     
    save_samples(samples, SAMPLES_OUTPUT_FILE_NAME)