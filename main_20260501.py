# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
selected = 1

data_list = {1:  {'name': '1_closed_shots_250RPM_50ml', 'rpm': 250, 'start': 40, 'stop': 45, 
                  'calculated': [52.92, 50.20, 50.17, 50.04, 49.88, 50.07, 49.88, 49.95], 
                  'target_next_round': [47.24, 47.05, 46.89, 46.86, 46.97, 46.91, 47.02, 47.06], 
                  'weightscale': [53, 103, 154, 204, 254, 305, 355, 405]}, 
             2:  {'name': '1_closed_shots_500RPM_50ml', 'rpm': 500, 'start': 35, 'stop': 95, 
                  'calculated': [56.10, 50.75, 49.82, 50.23, 51.92, 48.62, 49.99, 49.85], 
                  'target_next_round': [44.56, 43.90, 44.06, 43.86, 42.24, 43.44, 43.45, 43.57],
                  'weightscale': [56, 107, 157, 208, 258, 307, 357, 407]}, 
             3:  {'name': '1_closed_shots_750RPM_50ml', 'rpm': 750, 'start': 8, 'stop': 67, 
                  'calculated': [60.83, 51.04, 51.70, 50.07, 49.33, 49.91, 50.56, 49.32], 
                  'target_next_round': [41.10, 40.26, 38.93, 38.87, 39.40, 39.48, 39.04, 39.58], 
                  'weightscale': [60, 111, 162, 211, 260, 310, 359, 409]}, 
             4:  {'name': '1_closed_shots_1000RPM_50ml', 'rpm': 1000, 'start': 11, 'stop': 71, 
                  'calculated': [63.30, 54.22, 51.54, 49.81, 49.45, 49.80, 49.80, 50.93], 
                  'target_next_round': [39.49, 36.42, 35.33, 35.46, 35.86, 36.01, 36.15, 35.49],
                  'weightscale': [66, 119, 170, 220, 269, 320, 370, 420]},
            }
error_band = 1 # %
length = data_list.__len__()
fig, axes = plt.subplots(
    nrows=length,
    ncols=1,
    figsize=(10, 3 * length),
    sharex=True,
    constrained_layout=True
)
plt.subplot(length, 1, 1)
for key, value in data_list.items():
    plt.subplot(length, 1, key)
    # Fill aread betweeon two values
    plt.plot(range(len(value['calculated'])), 50*np.ones(len(value['calculated'])), '--', color='tab:gray', alpha=0.75, label=r'$V_{Target} = 50ml$')
    plt.fill_between(range(len(value['calculated'])), 50-(50*error_band/100), 50+(50*error_band/100), color='tab:orange', alpha=0.25, label=f'{error_band}% band')
    plt.scatter(range(len(value['calculated'])), value['calculated'], marker='o', s=35, edgecolors='darkred', facecolor='red', label=r'$V_{SF} \in \mathbb{R}$')
    plt.plot(range(len(value['calculated'])), value['calculated'], '-', color='red', alpha=0.25, linewidth=2)
    # plt.scatter(range(len(value['weightscale'])), [value['weightscale'][0]] + list(np.diff(value['weightscale'])), color='tab:blue', marker='o', label=r'$V_{Scale}$',)
    plt.vlines(
        range(len(value['weightscale'])),
        np.array([value['weightscale'][0]] + list(np.diff(value['weightscale']))) - 0.5,
        np.array([value['weightscale'][0]] + list(np.diff(value['weightscale']))) + 0.5,
        color='tab:blue',
        linewidth=4,
        label=r'$V_{Scale} \in \mathbb{Z}$'
    )
    # plt.plot(range(len(value['target_next_round'])), [50] + list(value['target_next_round'][0:-1]), marker='x', label=r'$V_{Target}$')
    plt.title(f'{value["rpm"]} RPM')
    plt.ylim(48, 67.5)
    plt.ylabel('Volume [ml]')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
plt.xlabel('Shot Number')
plt.suptitle("Closed-loop control of shot dosing at different RPMs", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/pdf_20260501/closed_loop_control_shots.pdf', format='pdf', bbox_inches='tight')
plt.show()