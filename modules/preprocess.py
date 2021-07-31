import pandas as pd
import numpy as np


def preprocess(train_store, test):
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
    return features_df, targets
