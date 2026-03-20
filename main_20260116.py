# %%
import pandas as pd

def load_data(filename, length_s=None):
    # Only import clumn 1, 2 and 6 
    data = pd.read_csv('data_logging_20260116/' + filename + '.txt', header=None,  usecols=[0, 1, 5], dtype={0: 'int64', 1: 'float64', 5: 'float64'})
    data.columns = ['timestamp_us', 'pressure_upstream', 'pressure_downstream']
    data['timestamp_us'] = data['timestamp_us'] - data['timestamp_us'].iloc[0]  # Normalize timestamp to start from zero

    # Keep the first 0.2 seconds of data
    if length_s is not None:
        data = data[data['timestamp_us'] <= length_s * 1e6]

    # Display the first few rows of the dataframe
    # print(data.head())

    return data 

# %%
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Roboto", "Helvetica Neue", "DejaVu Sans"],

})

def plot_different_rpms(filename, title, mean_subtracted=False, ylim=None):
    rpm_vec = [250, 500, 750, 1000]

    fig, axes = plt.subplots(1, len(rpm_vec), figsize=(12.5, 5), sharey=True)

    for ax, rpm in zip(axes, rpm_vec):
        data = load_data(f'{filename}_{rpm}RPM', length_s=0.25)
        if mean_subtracted:
            data['pressure_upstream'] -= data['pressure_upstream'].mean()
            data['pressure_downstream'] -= data['pressure_downstream'].mean()
        time_s = data['timestamp_us'] * 1e-6

        ax.plot(time_s, data['pressure_upstream'],
                color='blue', linewidth=1,
                label='Upstream')
        ax.plot(time_s, data['pressure_downstream'],
                color='red', linewidth=1,
                label='Downstream')
        ax.plot(time_s, data['pressure_upstream'],
                '.', color='blue', markersize=2,
                label='Upstream point')
        ax.plot(time_s, data['pressure_downstream'],
                '.', color='red', markersize=2,
                label='Downstream point')

        ax.set_title(f'{rpm} RPM ({rpm/60:.2f} Hz)')
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, round(max(time_s), 2))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Common labels & title
    if mean_subtracted:
        fig.supylabel(r'Pressure fluctuation [bar]')

    else:
        fig.supylabel('Pressure [bar]')
    fig.suptitle(title, fontsize=16)
    if ylim is not None:
        axes[0].set_ylim(ylim)

    # One legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.925)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.925])
    import os
    os.makedirs('figures/png_20260116', exist_ok=True)
    os.makedirs('figures/pdf_20260116', exist_ok=True)
    if mean_subtracted:
        plt.savefig(f'figures/png_20260116/{filename}_mean_subtracted.png', dpi=300)
        plt.savefig(f'figures/pdf_20260116/{filename}_mean_subtracted.pdf')
    else:
        plt.savefig(f'figures/png_20260116/{filename}.png', dpi=300)
        plt.savefig(f'figures/pdf_20260116/{filename}.pdf')
    plt.show()
    

plot_different_rpms('1_parallel_open', title='Sensors in Parallel Without Equilibar: Raw Pressure Signals at Different RPMs', ylim=[0, 3.0])
plot_different_rpms('1_parallel_open', title='Sensors in Parallel Without Equilibar: Mean Removed Pressure Signals at Different RPMs', mean_subtracted=True, ylim=[-0.25, 0.25])

plot_different_rpms('2_serie_open', title='Sensors in Serie Without Equilibar: Raw Pressure Signals at Different RPMs', ylim=[0, 3.0])
plot_different_rpms('2_serie_open', title='Sensors in Serie Without Equilibar: Mean Removed Pressure Signals at Different RPMs', mean_subtracted=True, ylim=[-0.25, 0.25])

# plot_different_rpms('3_serie_partial', title='Parallel Sensors With Equilibar: Raw Pressure Signals at Different RPMs', ylim=[0, 2.2])
# plot_different_rpms('3_serie_partial', title='Parallel Sensors With Equilibar: Mean Removed Pressure Signals at Different RPMs', mean_subtracted=True, ylim=[-0.25, 0.25])

plot_different_rpms('4_eq_no_ref_open', title='Sensors in Serie With Equilibar: Raw Pressure Signals at Different RPMs', ylim=[0, 3.0])
plot_different_rpms('4_eq_no_ref_open', title='Sensors in Serie With Equilibar: Mean Removed Pressure Signals at Different RPMs', mean_subtracted=True, ylim=[-0.25, 0.25])

pressure_vec = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
for pressure in pressure_vec:
    plot_different_rpms(f'6_eq_{pressure}mbar_open', title=f'Sensors in Serie With Equilibar at {pressure} mbar backpressure: Raw Pressure Signals at Different RPMs', ylim=[0, 3.0])
    plot_different_rpms(f'6_eq_{pressure}mbar_open', title=f'Sensors in Serie With Equilibar at {pressure} mbar backpressure: Mean Removed Pressure Signals at Different RPMs', mean_subtracted=True, ylim=[-0.25, 0.25])

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_signal_amplitudes(signal, fft_freq, target_freq=16.667, tol=2):
    """
    Compute amplitude of the dominant frequency near target_freq
    and the 4x peak around it.
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_vals = fft_vals[:n // 2 + 1]
    amplitude = np.abs(fft_vals) * 2 / n

    # Mask frequencies within tolerance of target frequency
    mask = (fft_freq >= target_freq - tol) & (fft_freq <= target_freq + tol)
    if not np.any(mask):
        raise ValueError("No frequency found near target frequency")

    dominant_idx = np.argmax(amplitude[mask])
    fft_indices = np.where(mask)[0]
    dominant_idx = fft_indices[dominant_idx]
    dominant_freq = fft_freq[dominant_idx]
    dominant_amp = amplitude[dominant_idx]

    # 4x dominant frequency
    freq_4x = 4 * dominant_freq
    mask_4x = (fft_freq >= freq_4x - tol) & (fft_freq <= freq_4x + tol)
    fft_indices_4x = np.where(mask_4x)[0]
    idx_peak_4x = fft_indices_4x[np.argmax(amplitude[mask_4x])]
    peak_amp_4x = amplitude[idx_peak_4x]

    return dominant_freq, dominant_amp, peak_amp_4x


def get_amplitudes(filename, rpm=1000):
    mapping = {'6_eq_-250mbar_open': '2_serie_open',
               '6_eq_0mbar_open': '4_eq_no_ref_open'}
    filename = mapping.get(filename, filename)
    data = load_data(f'{filename}_{rpm}RPM')

    signal_up = data['pressure_upstream'].values - np.mean(data['pressure_upstream'].values)
    signal_down = data['pressure_downstream'].values - np.mean(data['pressure_downstream'].values)
    time = data['timestamp_us'].values * 1e-6
    dt = np.mean(np.diff(time))
    n = len(signal_up)
    fft_freq = np.fft.fftfreq(n, dt)[:n // 2 + 1]

    # Pick dominant frequency near 1000 RPM
    dominant_freq_up, dominant_amp_up, peak_amp_4x_up = get_signal_amplitudes(signal_up, fft_freq, target_freq=rpm/60)
    dominant_freq_down, dominant_amp_down, peak_amp_4x_down = get_signal_amplitudes(signal_down, fft_freq, target_freq=rpm/60)

    # print(f"First frequency upstream: {dominant_freq_up:.2f} Hz")
    # print(f"First frequency downstream: {dominant_freq_down:.2f} Hz")

    # Get RMS values
    rms_up = np.sqrt(np.mean(signal_up**2))
    rms_down = np.sqrt(np.mean(signal_down**2))

    return dominant_amp_up, dominant_amp_down, peak_amp_4x_up, peak_amp_4x_down, rms_up, rms_down


# Collect results
rpm_vec = [250, 500, 750, 1000]

for rpm in rpm_vec:
    pressure_vec = [-250, 0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    result = [[p] + list(get_amplitudes(f'6_eq_{p}mbar_open', rpm=rpm)) for p in pressure_vec]

    # Determine mean amplitudes without Equilibar
    data = load_data(f'2_serie_open_{rpm}RPM')
    data_up = data['pressure_upstream'].values.mean()
    data_down = data['pressure_downstream'].values.mean()

    # Plot results
    result_df = pd.DataFrame(result, columns=['Backpressure [mbar]', 'Dominant Amp Upstream', 'Dominant Amp Downstream',
                                            '4x Freq Amp Upstream', '4x Freq Amp Downstream', 'RMS Upstream', 'RMS Downstream'])


    pressure_vec[0] = 'No Equilibar'  # Label for first entry
    x = np.arange(len(pressure_vec))  # index-based x-axis

    plt.figure(figsize=(10, 6))
    plt.plot(x, result_df['Dominant Amp Upstream'], 'bo-',
            label=f'Upstream: magnitude at {rpm/60:.01f} Hz')
    plt.plot(x, result_df['Dominant Amp Downstream'], 'ro-',
            label=f'Downstream: magnitude at {rpm/60:.01f} Hz')
    plt.plot(x, result_df['4x Freq Amp Upstream'], 'b--o',
            label=f'Upstream: magnitude at 4×{rpm/60:.01f} Hz')
    plt.plot(x, result_df['4x Freq Amp Downstream'], 'r--o',
            label=f'Downstream: magnitude at 4×{rpm/60:.01f} Hz')
    plt.axvline(1 + data_up*1000/250, color='blue', linestyle=':', label='Upstream w/o Equilibar')
    plt.axvline(1 + data_down*1000/250, color='red', linestyle=':', label='Downstream w/o Equilibar')
    plt.xticks(x, pressure_vec)  # <-- labels only
    plt.xlabel('Backpressure [mbar]')
    plt.ylabel('Frequency Magnitude [bar]')
    plt.title(f'Frequency Magnitude vs Backpressure at {rpm} RPM')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'figures/amplitude_vs_backpressure_{rpm}RPM.png', dpi=300)
    # plt.savefig(f'figures/amplitude_vs_backpressure_{rpm}RPM.pdf')
    plt.show()

    # % Plot the RMS values side-by-side
    bar_width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, result_df['RMS Upstream'], color='blue', alpha=0.7, width=bar_width,
            label='Upstream: RMS')
    plt.bar(x + bar_width/2, result_df['RMS Downstream'], color='red', alpha=0.7, width=bar_width,
            label='Downstream: RMS')
    plt.axvline(1 + data_up*1000/250, color='blue', linestyle=':', label='Upstream w/o Equilibar')
    plt.axvline(1 + data_down*1000/250, color='red', linestyle=':', label='Downstream w/o Equilibar')
    plt.xticks(x, pressure_vec)  # <-- labels only
    plt.xlabel('Backpressure [mbar]')
    plt.ylabel('RMS Value [bar]')
    plt.ylim(0, 0.16)
    plt.title(f'RMS Value vs Backpressure at {rpm} RPM')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'figures/rms_vs_backpressure_{rpm}RPM.png', dpi=300)
    # plt.savefig(f'figures/rms_vs_backpressure_{rpm}RPM.pdf')
    plt.show()

# %%
fig, axes = plt.subplots(
    nrows=len(rpm_vec),
    ncols=1,
    figsize=(10, 3 * len(rpm_vec)),
    sharex=True,
    sharey=True
)

for ax, rpm in zip(axes, rpm_vec):
    pressure_vec = [-250, 0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    result = [[p] + list(get_amplitudes(f'6_eq_{p}mbar_open', rpm=rpm)) for p in pressure_vec]

    data = load_data(f'2_serie_open_{rpm}RPM')
    data_up = data['pressure_upstream'].values.mean()
    data_down = data['pressure_downstream'].values.mean()

    df = pd.DataFrame(result, columns=[
        'Backpressure', 'DomUp', 'DomDown',
        '4xUp', '4xDown', 'RMS Up', 'RMS Down'
    ])

    x = np.arange(len(pressure_vec))
    pressure_labels = pressure_vec.copy()
    pressure_labels[0] = 'No Equilibar'

    bar_width = 0.35
    ax.bar(x - bar_width/2, df['RMS Up'], width=bar_width, label='Upstream', color='blue')
    ax.bar(x + bar_width/2, df['RMS Down'], width=bar_width, label='Downstream', color='red')

    ax.axvline(1 + data_up * 1000 / 250, color='blue', linestyle=':', label='Upstream pressure w/o Equilibar')
    ax.axvline(1 + data_down * 1000 / 250, color='red', linestyle=':', label='Downstream pressure w/o Equilibar')

    ax.set_title(f'{rpm} RPM', fontweight='bold')
    ax.grid(True)

    ax.set_ylabel('RMS [bar]')

    # Print RMS change in percentages compared to last value
    final = -1
    rms_up_no_eq = df['RMS Up'].iloc[0]
    rms_down_no_eq = df['RMS Down'].iloc[0]
    rms_up_final = df['RMS Up'].iloc[final]
    rms_down_final = df['RMS Down'].iloc[final]
    perc_change_up = (rms_up_final - rms_up_no_eq) / rms_up_no_eq * 100
    perc_change_down = (rms_down_final - rms_down_no_eq) / rms_down_no_eq * 100
    print(f'RMS change at {rpm} RPM Upstream: {perc_change_up:.0f}%, Downstream: {perc_change_down:.0f}%')

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(pressure_labels)
axes[-1].set_xlabel('Backpressure [mbar]')
axes[0].legend()
ax.set_ylim(0, 0.16)
fig.suptitle('RMS of Pressure Fluctuations vs Backpressure for Different RPMs', fontsize=16)

plt.tight_layout()
plt.show()