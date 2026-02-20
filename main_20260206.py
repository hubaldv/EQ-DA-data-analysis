# %%
def load_data(filename, start_s=None, stop_s=None):
    # Only import clumn 1, 2 and 6 
    data = pd.read_csv('data_logging_20260206/' + filename + '.txt', header=None,  usecols=[0, 1, 5], dtype={0: 'int64', 1: 'float64', 5: 'float64'})
    data.columns = ['timestamp_us', 'pressure_equilibar', 'pressure_process']
    data['timestamp_us'] = data['timestamp_us'] - data['timestamp_us'].iloc[0]  # Normalize timestamp to start from zero

    # data = data[(data['timestamp_us'] >= 6 * 1e6) & (data['timestamp_us'] <= 7 * 1e6)]
    # Keep the first 0.2 seconds of data
    if start_s is not None:
        data = data[data['timestamp_us'] >= start_s * 1e6]
    if stop_s is not None:
        data = data[data['timestamp_us'] <= stop_s * 1e6]

    # Display the first few rows of the dataframe
    # print(data.head())

    return data 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a dictionary to store the data for different files and conditions
data_list =    {'2_250RPM_1_5ref':  {'enabled': [2, 4], 'disabled': [13, 15],   'rpm': 250,  'ref': 1.5, 'length': [0, 20]}, 
                '2_500RPM_1_5ref':  {'enabled': [2, 4], 'disabled': [10, 11],   'rpm': 500,  'ref': 1.5, 'length': [0, 12]}, 
                '2_750RPM_1_5ref':  {'enabled': [2, 4], 'disabled': [6.4, 6.9],  'rpm': 750,  'ref': 1.5, 'length': [0, 9]}, 
                '2_1000RPM_1_5ref': {'enabled': [2, 4], 'disabled': [7.5, 8.5], 'rpm': 1000, 'ref': 1.5, 'length': [0, 10]}}

data_list_keys = list(data_list.keys())
key_selected = data_list_keys[0]
print(key_selected)

data = load_data(key_selected)
time_s = data['timestamp_us'] / 1e6  # convert to seconds
pressure_equilibar = data['pressure_equilibar']
pressure_process = data['pressure_process']

fig, axes = plt.subplots(1, len(data_list_keys), figsize=(12.5, 5), sharey=True)

for ax, key in zip(axes.flatten(), data_list_keys):
    data = load_data(key, data_list[key]['length'][0], data_list[key]['length'][1])
    time_s = data['timestamp_us'] / 1e6
    pressure_equilibar = data['pressure_equilibar']
    pressure_process = data['pressure_process']

    ax.plot(time_s, pressure_equilibar, label='Upstream Equilibar', color='orange', alpha=0.75)
    ax.plot(time_s, pressure_process, label='Upstream process', color='blue', alpha=0.75)
    
    ax.set_ylim(0.8, 1.25)

    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax],
                     data_list[key]['enabled'][0],
                     data_list[key]['enabled'][1],
                     color='green', alpha=0.3, label='Equilibar enabled')

    ax.fill_betweenx([ymin, ymax],
                     data_list[key]['disabled'][0],
                     data_list[key]['disabled'][1],
                     color='red', alpha=0.3, label='Equilibar disabled')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_title(f"{data_list[key]['rpm']} RPM")
    ax.grid()

plt.suptitle('Pressures vs Time for different RPMs at 1.5 bar reference pressure', fontsize=16, fontweight='bold')
handles, labels = axes.flatten()[0].get_legend_handles_labels()

fig.legend(
        handles, labels,
        loc='upper center',
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.925)
    )
plt.tight_layout(rect=[0, 0, 1, 0.925])
plt.savefig('data_logging_20260213/pressure_vs_time_1_5ref.eps', format='eps')
plt.show()

# Save as eps

# %%
def rms_for_period(data, start_s, stop_s):
    period_data = data[(data['timestamp_us'] >= start_s*1e6) & (data['timestamp_us'] <= stop_s*1e6)]
    rms_equilibar = np.sqrt(np.mean((period_data['pressure_equilibar'] - period_data['pressure_equilibar'].mean())**2))
    rms_process = np.sqrt(np.mean((period_data['pressure_process'] - period_data['pressure_process'].mean())**2))
    return rms_equilibar, rms_process

def calculate_rms(data, key_selected, data_list):
    enabled = data_list[key_selected]['enabled']
    disabled = data_list[key_selected]['disabled']
    rms_enabled = rms_for_period(data, *enabled)
    rms_disabled = rms_for_period(data, *disabled)
    return *rms_enabled, *rms_disabled

# Plot bars
rpms = []
enabled_rms_process = []
disabled_rms_process = []
for key in data_list_keys:
    data = load_data(key)
    # Compute RMS
    rms_enabled_equilibar, rms_enabled_process, rms_disabled_equilibar, rms_disabled_process = calculate_rms(data, key, data_list)
    # Print in table
    print(f"{key:<20} {'RMS at Equilibar':<20} {'RMS at Process':<20}")
    print(f"{'Equilibar Enabled':<20} {rms_enabled_equilibar:<20.4f} {rms_enabled_process:<20.4f}")
    print(f"{'Equilibar Disabled':<20} {rms_disabled_equilibar:<20.4f} {rms_disabled_process:<20.4f}")
    print(f"Reduction in RMS: {(rms_enabled_process - rms_disabled_process)/rms_disabled_process*100:.1f}%")
    rpms.append(data_list[key]['rpm'])
    enabled_rms_process.append(rms_enabled_process)
    disabled_rms_process.append(rms_disabled_process)

# Bar width and positions
bar_width = 0.25
x = np.arange(len(rpms))

plt.figure(figsize=(12.5, 5))
plt.bar(x - 0.5*bar_width, disabled_rms_process, width=bar_width, label='Equilibar Disabled', color='red', alpha=0.7)
plt.bar(x + 0.5*bar_width, enabled_rms_process, width=bar_width, label='Equilibar Enabled', color='green', alpha=0.7)

plt.xlabel('RPM')
plt.ylabel('RMS Pressure [bar]')
plt.title('RMS Pressure vs RPM', fontsize=16, fontweight='bold')
plt.xticks(x, rpms)  # set RPM labels at correct positions
plt.legend()
plt.grid(True)

plt.savefig('data_logging_20260213/rms_pressure_vs_rpm.eps', format='eps')
plt.show()