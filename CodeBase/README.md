Steps to run the Project to reproduce results:

1. Setup the environment using command `pip install -r requirements.txt`
2. Run the `pre_process.ipynb` notebook to pre-process and encode the input dataset and make it ready for training our model.
3. `utils.py` includes helper function for unsupervised anomaly detection, `sequential_forward_selection.ipynb`  includes the code and performance of multiple unsupervised anomaly detection on valid data based on our greedy search approach.
4. `RFOD.ipynb` is used to run the RFOD framework and get performance results on the validation dataset and generate the submission csv file on the unseen test dataset. This file also saves the RFOD model we have trained.
5. `rfod_model1.joblib` and `rfod_model2.joblib` are the models we have trained and saved as part of running the `RFOD.ipynb` notebook to ensure we do not have to retrain the model each time.

Data Folder includes the below dataset files:

1. `loan_train.csv`, `loans_valid.csv` and `loans_test.csv`, are the Freddie Mac Loan Level Dataset used by us for our project.
2. `encode_train.csv`, `encode_valid.csv` and `encode_test.csv` are the pre-processed and encoded dataset geenrated by `pre_process.ipynb` notebook, which is then used to train our models and validate their performance.
3. `sub_rfod1.csv` and `sub_rfod2.csv` are the anomaly scores generated on test dataset by our RFOD models when trained with different hyper-parameters.
4. `sub_ensemble.csv` is the final ensembled anomaly score we submitted on Kaggle. (FINAL SUBMISSION FILE ON KAGGLE)
