To run the LSTM code, first extract the features using the TSN trained models

For example, train and test using the `expts/003_CATER_loc_RGB.sh` and `expts/003_CATER_loc_RGB_test.sh`. Then, train and test the LSTM using `expts/003_CATER_loc_RGB_lstm.sh`. The test file will need to be run twice, once for train and once for test. Just edit the .sh file (change the --split) and run it again.
