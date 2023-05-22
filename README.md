# cse493w-tesla-charge-port-project
Repo for my CSE493W project, which records Tesla's "open charge port" radio signal and rebroadcasts it using a HackRF to open Tesla charge ports.

It also contains code I wrote where I attempted to replay Tesla app BLE packets to unlock a car.

- record_charger_radio.py - records the "open Tesla charge port" signal + saves it.

- record_ble.py - records BLE packets on the 2.45GHz frequency.

- record_common.py - contains code shared across the record_* files.

- send_charger_radio.py - rebroadcasts the saved "open Tesla charge port" signal.

- record_ble.py - records BLE packets on the 2.45GHz frequency.
  - NOTE:  Although this code _does_ successfully rebroadcast BLE packets, they expire before they can be successfully used to unlock a Tesla. 

- send_common.py - contains code shared across the send_* files.