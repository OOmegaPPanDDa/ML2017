#!/bin/bash

# python3 hw5_best_train.py train_data.csv test_data.csv

wget https://www.dropbox.com/s/xc9p8y3eiu2q3z5/best_bag_complete_model0.h5?dl=1 -O best_bag_complete_model0.h5
wget https://www.dropbox.com/s/yj864uwavii5pv3/best_bag_complete_model1.h5?dl=1 -O best_bag_complete_model1.h5
wget https://www.dropbox.com/s/rloiain38bo06zz/best_bag_complete_model2.h5?dl=1 -O best_bag_complete_model2.h5
wget https://www.dropbox.com/s/r337hxrmyclhl7a/best_bag_complete_model3.h5?dl=1 -O best_bag_complete_model3.h5
wget https://www.dropbox.com/s/etzvq49up627wef/best_bag_complete_model4.h5?dl=1 -O best_bag_complete_model4.h5
wget https://www.dropbox.com/s/dkxhrxo08adf5rz/best_bag_complete_model5.h5?dl=1 -O best_bag_complete_model5.h5
wget https://www.dropbox.com/s/y8syhzxopaadb44/best_bag_complete_model6.h5?dl=1 -O best_bag_complete_model6.h5
wget https://www.dropbox.com/s/9rze6ho5pgf2r3d/best_bag_complete_model7.h5?dl=1 -O best_bag_complete_model7.h5
wget https://www.dropbox.com/s/zm22ojn71qaqu16/best_bag_complete_model8.h5?dl=1 -O best_bag_complete_model8.h5
wget https://www.dropbox.com/s/gljqa4gmw4qxo7b/best_bag_complete_model9.h5?dl=1 -O best_bag_complete_model9.h5

wget https://www.dropbox.com/s/balgh62b5xnfeav/best_rnn_complete_model0.h5?dl=1 -O best_rnn_complete_model0.h5
wget https://www.dropbox.com/s/gr2ejowv67h3pm6/best_rnn_complete_model1.h5?dl=1 -O best_rnn_complete_model1.h5
wget https://www.dropbox.com/s/ljgp8w8sg0gtkps/best_rnn_complete_model2.h5?dl=1 -O best_rnn_complete_model2.h5
wget https://www.dropbox.com/s/k9d5qe7wobhjghs/best_rnn_complete_model3.h5?dl=1 -O best_rnn_complete_model3.h5
wget https://www.dropbox.com/s/edzfow1xclx0gwq/best_rnn_complete_model4.h5?dl=1 -O best_rnn_complete_model4.h5
wget https://www.dropbox.com/s/5i1gfccypyooehz/best_rnn_complete_model5.h5?dl=1 -O best_rnn_complete_model5.h5
wget https://www.dropbox.com/s/dcqmg3p7gbaukui/best_rnn_complete_model6.h5?dl=1 -O best_rnn_complete_model6.h5
wget https://www.dropbox.com/s/blai9ts5ry8zrte/best_rnn_complete_model7.h5?dl=1 -O best_rnn_complete_model7.h5
wget https://www.dropbox.com/s/nue4vkkpoq1mjye/best_rnn_complete_model8.h5?dl=1 -O best_rnn_complete_model8.h5
wget https://www.dropbox.com/s/khyctkx743t49eh/best_rnn_complete_model9.h5?dl=1 -O best_rnn_complete_model9.h5

python3 hw5_best_test.py $1 $2