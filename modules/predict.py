import pandas as pd
import numpy as np
import pickle

FEATURES = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday',
            'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']
df = pd.DataFrame({
    'Store': 55,
    'DayOfWeek': 4,
    'Open': 1,
    'Promo': 1,
    'StateHoliday': 0,
    'SchoolHoliday': 0,
    'Year': 2015,
    'Month': 9,
    'Day': 3,
    'WeekOfYear': 30,
    'Baba': 99
}, index=[0])


def make_prediction(df):
    loaded_model = pickle.load(open("../models/model.pkl", 'rb'))
    df = df[FEATURES]
    result = loaded_model.predict(df)
    print("RESULT:", np.exp(result))
    return np.exp(result)

