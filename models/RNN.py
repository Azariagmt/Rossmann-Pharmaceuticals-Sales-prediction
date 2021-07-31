import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

time_series_data = pd.read_csv('../rossmann-store-sales/train_sales.csv',
                               index_col="Date", parse_dates=True)
time_series_data = time_series_data['Sales']
time_series_data = pd.DataFrame({
    "Date": time_series_data.index,
    "Sales": time_series_data
})
time_series_data = time_series_data.set_index("Date")
print(time_series_data.head())

scaler = MinMaxScaler()
scaler.fit(time_series_data.Sales.values.reshape([-1, 1]))
sales_scaled = scaler.transform(time_series_data.Sales.values.reshape(-1, 1))
time_series_data['SalesScaled'] = sales_scaled
print(time_series_data.tail(10))

print(time_series_data.describe())
fig = plt.figure()
gs = GridSpec(2, 1, figure=fig)

fig.set_figheight(20)
fig.set_figwidth(30)
fig.tight_layout(pad=15)

M = 100
xticks = ticker.MaxNLocator(M)

ax1 = fig.add_subplot(gs[0,0])
ax1.plot(time_series_data.index, time_series_data.Sales, 'b-')
ax1.xaxis.set_major_locator(xticks)
ax1.tick_params(labelrotation=90)
ax1.set_xlabel('Date')
ax1.set_ylabel('Thousands of Units')
ax1.title.set_text('Time Series Plot of Sales')
ax1.grid(True)

ax2 = fig.add_subplot(gs[1,0])
ax2.plot(time_series_data.index, time_series_data.SalesScaled, 'g-')
ax2.xaxis.set_major_locator(xticks)
ax2.tick_params(labelrotation=90)
ax2.set_xlabel('Date')
ax2.set_ylabel('Scaled Units')
ax2.title.set_text('Time Series Plot of Min Max Scaled Sales numbers')
ax2.grid(True)
plt.show()

fig = plt.figure()
gs = GridSpec(2, 1, figure=fig)

fig.set_figheight(10)
fig.set_figwidth(30)
fig.tight_layout(pad=6)

ax1 = fig.add_subplot(gs[0,0])
ax1.hist(time_series_data.Sales, density=True, bins=60)
ax1.title.set_text('Histogram  Sales')
ax1.grid(True)

ax2 = fig.add_subplot(gs[1,0])
ax2.hist(time_series_data.SalesScaled, density=True, bins=60)
ax2.title.set_text('Histogram of the of Min Max Scaled  Sales')
ax2.grid(True)
plt.show()

adfResult = adfuller(time_series_data.Sales.values, autolag='AIC')
print(f'ADF Statistic: {adfResult[0]}')
print(f'p-value: {adfResult[1]}')