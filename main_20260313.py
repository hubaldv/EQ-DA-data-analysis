# %%
# If folder not exists, create it
import os
if not os.path.exists('figures/pdf_20260313'):
    os.makedirs('figures/pdf_20260313')

def load_data(filename, start_s=None, stop_s=None):
    # Only import clumn 1, 2 and 6 
    data = pd.read_csv('data_logging_20260313/' + filename + '.txt', header=None,  usecols=[0, 1, 3, 5, 6, 7, 8], dtype={0: 'int64', 1: 'float64', 5: 'float64'})
    data.columns = ['timestamp_us', 'reference_pressure', 'setpoint', 'temperature', 'valve_a', 'valve_b', 'flow']
    data['timestamp_us'] = data['timestamp_us'] - data['timestamp_us'].iloc[0]  # Normalize timestamp to start from zero

    # data = data[(data['timestamp_us'] >= 6 * 1e6) & (data['timestamp_us'] <= 7 * 1e6)]
    # Keep the first 0.2 seconds of data
    if start_s is not None:
        data = data[data['timestamp_us'] >= start_s * 1e6]
    if stop_s is not None:
        data = data[data['timestamp_us'] <= stop_s * 1e6]

    data['timestamp_us'] = data['timestamp_us'] - data['timestamp_us'].iloc[0]  # Re-normalize timestamp to start from zero after filtering
    # Display the first few rows of the dataframe
    # print(data.head())

    return data 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

selected = 1
data_list = {1:  {'name': '1_shot_dosing_250RPM', 'rpm': 250, 'start': 38, 'stop': 95, 'volume': [30, 30, 30, 30]}, 
             2:  {'name': '1_shot_dosing_500RPM', 'rpm': 500, 'start': 35, 'stop': 95, 'volume': [48, 48, 48, 47]}, 
             3:  {'name': '1_shot_dosing_750RPM', 'rpm': 750, 'start': 8, 'stop': 67, 'volume': [64, 64, 64, 64]}, 
             4:  {'name': '1_shot_dosing_1000RPM', 'rpm': 1000, 'start': 11, 'stop': 71, 'volume': [105, 108, 105, 106]},
             5:  {'name': '1_shot_dosing_250RPM', 'rpm': 250, 'start': 40, 'stop': 48, 'volume': []}}

data = load_data(f'{data_list[selected]["name"]}', start_s=data_list[selected]['start'], stop_s=data_list[selected]['stop'])

plt.figure(figsize=(10, 9))
plt.subplot(3, 1, 1)
plt.plot(data['timestamp_us'] / 1e6, data['setpoint'], color='tab:gray', linestyle='--', label='Setpoint', alpha=0.75)
plt.plot(data['timestamp_us'] / 1e6, data['reference_pressure'], color='tab:orange', label='Measured Pressure', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.title('Reference Pressure')
plt.legend(loc='upper right')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(data['timestamp_us'] / 1e6, data['valve_a']/20000*100, color='tab:blue', label='Valve A', alpha=0.75)
plt.plot(data['timestamp_us'] / 1e6, data['valve_b']/20000*100, color='tab:red', label='Valve B', alpha=0.75)
plt.xlabel('Time [s]')
plt.ylabel('Opening [%]')
plt.title('Valve action')
plt.legend(loc='upper right')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(data['timestamp_us'] / 1e6, data['flow'], color='tab:green', label='Flow')
plt.xlabel('Time [s]')
plt.ylabel('Flow [ml/min]')
plt.title('Flow')
plt.legend(loc='upper right')
plt.grid()
plt.suptitle(f"Shot dosing at {data_list[selected]['rpm']} RPM", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/pdf_20260313/open_loop_flow_overview_{data_list[selected]["rpm"]}RPM.pdf', bbox_inches='tight')
plt.show()

# %% Plot a zoomed in view of a single case
def get_data(selected):
    cfg = data_list[selected]
    return load_data(
        f'{data_list[selected]["name"]}',
        start_s=cfg['start'],
        stop_s=cfg['stop']
    ), cfg


def detect_events(data, threshold=6, gap_us=1e6):
    dosing = data[data['flow'] > threshold].copy()

    if dosing.empty:
        return dosing, pd.DataFrame(columns=['min', 'max'])

    dosing['event_id'] = (dosing['timestamp_us'].diff() > gap_us).cumsum()
    ranges = dosing.groupby('event_id')['timestamp_us'].agg(['min', 'max'])

    return dosing, ranges


def compute_event_volumes(data, event_ranges):
    volumes = []

    for _, row in event_ranges.iterrows():
        event = data[
            (data['timestamp_us'] >= row['min']) &
            (data['timestamp_us'] <= row['max'])
        ]

        t = event['timestamp_us'].to_numpy() / 1e6
        flow = event['flow'].to_numpy()

        # baseline correction + ml/min → ml/s
        flow = (flow - flow[0]) / 60

        vol = np.trapezoid(flow, t)
        volumes.append(vol)

        print(f"Event from {row['min']/1e6:.2f}s to {row['max']/1e6:.2f}s: Volume = {vol:.2f} ml")

    return volumes


def plot(selected, ax=None, threshold=6):
    if ax is None:
        ax = plt.gca()

    data, cfg = get_data(selected)

    # Precompute once
    t = data['timestamp_us'] / 1e6

    _, event_ranges = detect_events(data, threshold)

    # Plot signal
    ax.plot(t, data['flow'],
            color='tab:green', marker='.', markersize=4,
            linewidth=1, label='Flow')

    # First point highlight
    ax.plot(t.iloc[0], data['flow'].iloc[0],
            color='tab:red', marker='o', markersize=6)

    # Plot events safely
    span_handle = None
    for _, row in event_ranges.iterrows():
        span_handle = ax.axvspan(row['min']/1e6, row['max']/1e6,
                                color='tab:orange', alpha=0.3)

    # Labels
    ax.set(
        xlabel='Time [s]',
        ylabel='Flow [ml/min]',
        title=f'{cfg["rpm"]} RPM'
    )

    # Legend (safe)
    handles, labels = ax.get_legend_handles_labels()
    if span_handle is not None:
        handles.append(span_handle)
        labels.append('Dosing event')

    ax.legend(handles, labels, loc='upper right')
    ax.grid()

    # Compute volumes
    print(f"\nComputing volumes for {cfg['rpm']} RPM:")
    volumes = compute_event_volumes(data, event_ranges)
    # Print standard deviation of volumes
    if len(volumes) > 1:
 
        v_scale = data_list[selected]['volume']
        for i, vol in enumerate(volumes):
            deviation = (vol - v_scale[i]) / v_scale[i] * 100
            print(f"Event {i+1}: Volume = {vol:.2f} ml (Deviation: {deviation:.2f}%)")

        print(f"Standard deviation of VSF and v_scale: {np.std(volumes):.2f} ml, {np.std(v_scale):.2f} g")

    total_volume = sum(volumes)

    return total_volume, volumes

selected = 5
plt.figure(figsize=(12, 6))
plot(selected)
plt.title(f"Zoomed in view of flow for {data_list[selected]['rpm']} RPM", fontsize=14, fontweight='bold')
plt.savefig(f'figures/pdf_20260313/open_loop_flow_zoomed_{data_list[selected]["rpm"]}RPM.pdf', bbox_inches='tight')
plt.show()

# % Plot all cases together
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

axes = axes.flatten()

for i in range(4):
    plot(i + 1, ax=axes[i])

plt.suptitle('Flow vs Time for different RPMs', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/pdf_20260313/open_loop_flow_all_cases.pdf', format="pdf", dpi=100, bbox_inches="tight")
plt.savefig('figures/pdf_20260313/open_loop_flow_all_cases.png', format="png", dpi=100, bbox_inches="tight")
plt.show()

# %%
selected = 1
data_list = {1:  {'name': '2_shot_dosing_250RPM_50ml_2', 'rpm': 250, 'start': 7, 'stop': 70, 'volume': [50, 50, 50, 51]}, 
             2:  {'name': '2_shot_dosing_500RPM_50ml', 'rpm': 500, 'start': 5, 'stop': 45, 'volume': [52, 52, 52, 52]}, 
             3:  {'name': '2_shot_dosing_750RPM_50ml', 'rpm': 750, 'start': 6, 'stop': 42, 'volume': [54, 55, 54, 55]}, 
             4:  {'name': '2_shot_dosing_1000RPM_50ml_2', 'rpm': 1000, 'start': 12, 'stop': 71, 'volume': [55, 56, 55, 56]}}

data = load_data(f'{data_list[selected]["name"]}', start_s=data_list[selected]['start'], stop_s=data_list[selected]['stop'])

plt.figure(figsize=(10, 9))
plt.subplot(3, 1, 1)
plt.plot(data['timestamp_us'] / 1e6, data['setpoint'], color='tab:gray', linestyle='--', label='Setpoint', alpha=0.75)
plt.plot(data['timestamp_us'] / 1e6, data['reference_pressure'], color='tab:orange', label='Measured Pressure', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.title('Reference Pressure')
plt.legend(loc='upper right')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(data['timestamp_us'] / 1e6, data['valve_a']/20000*100, color='tab:blue', label='Valve A', alpha=0.75)
plt.plot(data['timestamp_us'] / 1e6, data['valve_b']/20000*100, color='tab:red', label='Valve B', alpha=0.75)
plt.xlabel('Time [s]')
plt.ylabel('Opening [%]')
plt.title('Valve action')
plt.legend(loc='upper right')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(data['timestamp_us'] / 1e6, data['flow'], color='tab:green', label='Flow')
plt.xlabel('Time [s]')
plt.ylabel('Flow [ml/min]')
plt.title('Flow')
plt.legend(loc='upper right')
plt.grid()
plt.suptitle(f"Shot dosing at {data_list[selected]['rpm']} RPM", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/pdf_20260313/closed_loop_flow_overview_{data_list[selected]["rpm"]}RPM.pdf', bbox_inches='tight')
plt.show()

# %% Plot all cases together
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

axes = axes.flatten()

for i in range(4):
    plot(i + 1, ax=axes[i])

plt.suptitle('Flow vs Time for different RPMs', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/pdf_20260313/closed_loop_flow_all_cases.pdf', format="pdf", dpi=100, bbox_inches="tight")
plt.savefig('figures/pdf_20260313/closed_loop_flow_all_cases.png', format="png", dpi=100, bbox_inches="tight")
plt.show()


# %%
