#!/bin/bash

# python3 hw5_rnn_train.py train_data.csv

wget https://www.dropbox.com/s/t5mmbqht6c0czu3/rnn_complete_model.h5?dl=1 -O rnn_complete_model.h5

python3 hw5_rnn_test.py $1 $2