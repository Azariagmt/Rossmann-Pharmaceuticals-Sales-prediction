import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn import *
import logging
import numpy as np
import pandas as pd
import dvc.api
import mlflow
import mlflow.sklearn
import warnings
import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os
import logging
sys.path.insert(0, '../modules')

from logs import log
from preprocess import preprocess
# configures logger
logger = log(path="../logs/", file="rfr.logs")
logger.info("Starts RFR")

train_store_path = 'rossmann-store-sales/train_store.csv'
repo = "../"
version = "'trainstorev1'"

train_store_url = dvc.api.get_url(
    path=train_store_path,
    repo=repo
)

mlflow.set_experiment('Rossmann Pharmaceuticals sales price prediction')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(50)

    def rmspe(y, yhat):
        '''
        Loss function for model evaluation
        '''
        rmspe = np.sqrt(np.mean((y - yhat)**2))
        return rmspe

    train_store = pd.read_csv(
        '../rossmann-store-sales/train_store.csv', parse_dates=True, index_col=0)
    mlflow.log_param('train_store_data_url', train_store_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('model_type', 'Random Forest')

    test = pd.read_csv('../rossmann-store-sales/test.csv',
                       index_col="Date", parse_dates=True)

    # send to preprocess modules
    features_df, targets = preprocess(train_store, test)
    X_train, X_train_test, y_train, y_train_test = model_selection.train_test_split(
        features_df, targets, test_size=0.20, random_state=15)
    logger.info("Training and testing split was successful.")
    mlflow.log_param('input_rows', X_train.shape[0])
    mlflow.log_param('input_cols', X_train.shape[1])

    rfr = RandomForestRegressor(n_estimators=50,
                                criterion='mse',
                                max_depth=10,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='auto',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=4,
                                random_state=31,
                                verbose=0,
                                warm_start=False)
    rfr.fit(X_train, y_train)
    logger.info("Model fit successful")
    mlflow.sklearn.log_model(rfr, "random forest model")

    yhat = rfr.predict(X_train_test)
    error = rmspe(y_train_test, yhat)
    logger.info(f"error{str(error)}")
    mlflow.log_param('error', error)
