# %%
def load_data(filename, length_s=None):
    # Only import clumn 1, 2 and 6 
    data = pd.read_csv('data_logging_20260206/' + filename + '.txt', header=None,  usecols=[0, 1, 5], dtype={0: 'int64', 1: 'float64', 5: 'float64'})
    data.columns = ['timestamp_us', 'pressure_upstream', 'pressure_downstream']
    data['timestamp_us'] = data['timestamp_us'] - data['timestamp_us'].iloc[0]  # Normalize timestamp to start from zero

    # data = data[(data['timestamp_us'] >= 6 * 1e6) & (data['timestamp_us'] <= 7 * 1e6)]
    # Keep the first 0.2 seconds of data
    if length_s is not None:
        data = data[data['timestamp_us'] <= length_s * 1e6]

    # Display the first few rows of the dataframe
    # print(data.head())

    return data 

import matplotlib.pyplot as plt
import pandas as pd

data = load_data('1_250RPM_1_5ref', length_s=0.2)
data = load_data('1_323RPM_0_8ref', length_s=0.2)
data = load_data('knijper', length_s=0.2)
data = load_data('2_250RPM_3_0ref')
data = load_data('2_250RPM_1_5ref')
# data = load_data('2_500RPM_1_5ref')
# data = load_data('2_750RPM_1_5ref')
# data = load_data('2_1000RPM_1_5ref')
data = load_data('data')


plt.figure(figsize=(10, 6))
plt.plot(data['timestamp_us'] / 1e6, data['pressure_upstream'], label='Before Equilibar', color='blue', marker='.', markersize=4)
plt.plot(data['timestamp_us'] / 1e6, data['pressure_downstream'], label='Before preocess', color='red', marker='.', markersize=4)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.title('Pressure vs Time')
plt.legend()
plt.grid()
plt.show()
