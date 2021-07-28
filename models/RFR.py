import logging
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble import RandomForestRegressor
import pickle

import dvc.api
import mlflow
import mlflow.sklearn
import logging
import warnings

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


train_store_path = 'rossmann-store-sales/train_store.csv'
repo = "../"
version = "'trainstorev1'"

data_url = dvc.api.get_url(
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
    mlflow.log_param('train_store_data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', train_store.shape[0])
    mlflow.log_param('input_cols', train_store.shape[1])
    mlflow.log_param('model_type', 'Random Forest')

    test = pd.read_csv('../rossmann-store-sales/test.csv',
                       index_col="Date", parse_dates=True)

    # since competition open since have similar meanings we can merge into once
    train_store['CompetitionOpenSince'] = np.where((train_store['CompetitionOpenSinceMonth'] == 0) & (train_store['CompetitionOpenSinceYear'] == 0), 0, (train_store.Month - train_store.CompetitionOpenSinceMonth) +
                                                   (12 * (train_store.Year - train_store.CompetitionOpenSinceYear)))

    # we can get rid of `CompetitionOpenSinceYear` and `CompeitionOpenSinceMonth`
    del train_store['CompetitionOpenSinceYear']
    del train_store['CompetitionOpenSinceMonth']

    # data extraction
    # TODO: extract to sklearn pipelines
    test['Year'] = test.index.year
    test['Month'] = test.index.month
    test['Day'] = test.index.day
    test['WeekOfYear'] = test.index.weekofyear

    print(train_store.dtypes)
    # print(train_store[train_store['StateHoliday'].na()])
    # transform stateholiday
    train_store["StateHoliday"] = train_store['StateHoliday'].map(
        {"0": 0, "a": 1, "b": 1, "c": 1})
    features = test.columns.tolist()
    features.pop(0)
    features_df = train_store[features]
    print(features_df.head())
    targets = np.log(train_store.Sales)
    print(targets)
    # targets = float(targets)

    X_train, X_train_test, y_train, y_train_test = model_selection.train_test_split(
        features_df, targets, test_size=0.20, random_state=15)
    print("Training and testing split was successful.")

    rfr = RandomForestRegressor(n_estimators=10,
                                criterion='mse',
                                max_depth=5,
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

    filename = 'model1.sav'
    pickle.dump(rfr, open(filename, 'wb'))

    yhat = rfr.predict(X_train_test)
    error = rmspe(y_train_test, yhat)
    print(error)
    mlflow.log_param('error', error)
