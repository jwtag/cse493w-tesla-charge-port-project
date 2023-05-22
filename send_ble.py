from record_ble import FSPS, RADIO_FREQ, SAMPLES_OUTPUT_FILE_NAME
from send_common import send_samples

send_samples(SAMPLES_OUTPUT_FILE_NAME, FSPS, RADIO_FREQ)