Prepare
1. put dataset in test/data file
2. Run test/download_model.sh, unzip and rename the file as test/save

Run Order

(preprocess part)
1. use python3 to run test/preprocess.py
2. use python3 to run test/location_pred.py

(test part)
3. use python3 to run test/test.py
4. use python3 to run test/format.py
5. test/test_label.csv is the prediction result

Run Script (in test file)
1.	Run test/doall.sh