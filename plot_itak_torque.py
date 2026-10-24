"""Itak: mean right-ankle exo torque profile per slip level (torque-on trials).

The Vicon jet channel is flat in every Itak marker file, so the exo logs
cannot be aligned to Vicon / tread1. Time is instead aligned to right-foot
gyro onset, the same rule as plot_commanded_torque_subjects.py: the first
sample where |foot_right_gyro_z| exceeds 0.5 rad/s above the median of the
first 0.5 s.

Top row = commanded torque (torque_sent_R), bottom row = measured torque
(torque_measured_R). One column per slip level (thin = trials, bold = mean,
band = ±1 SD), plus a last column overlaying the slip means.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE, 'Itak', 'IMU')
WINDOW = (-0.35, 1.0)  # s relative to right-foot gyro onset
GYRO_THR = 0.5  # rad/s above the quiet-standing median
DT = 0.025
NAME = re.compile(r'Pert_Policy_trial_(\d+)_slip_(\d+)(_notorque)?_')
SLIP_COLORS = ['#6baed6', '#2171b5', '#08306b']  # light -> dark with slip level
ROWS = (('sent', 'Commanded torque, R (N·m)'), ('measured', 'Measured torque, R (N·m)'))


def load_trials():
    trials = []
    for path in sorted(glob.glob(os.path.join(FOLDER, 'Pert_Policy_trial_*_data.csv'))):
        m = NAME.search(os.path.basename(path))
        if m is None or m.group(3):
            continue
        d = pd.read_csv(path)
        ts = d['timestamp'].values
        gyro = d['foot_right_gyro_z'].values
        n_base = max(8, int(round(0.5 / np.median(np.diff(ts)))))
        moved = np.where(np.abs(gyro - np.median(gyro[:n_base])) > GYRO_THR)[0]
        if len(moved) == 0:
            print(f'  Trial {m.group(1)} slip {m.group(2)}: no gyro onset, skipped')
            continue
        trials.append(dict(trial=int(m.group(1)), slip=int(m.group(2)), t=ts - ts[moved[0]],
                           sent=d['torque_sent_R'].values, measured=d['torque_measured_R'].values))
    return trials


def mean_sd(group, key, grid):
    stack = np.vstack([np.interp(grid, t['t'], t[key], left=np.nan, right=np.nan) for t in group])
    sd = np.nanstd(stack, 0, ddof=1) if len(group) > 1 else np.full(grid.shape, np.nan)
    return np.nanmean(stack, 0), sd


def main():
    trials = load_trials()
    slips = sorted({t['slip'] for t in trials})
    colors = dict(zip(slips, SLIP_COLORS[-len(slips):]))
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)

    fig, axes = plt.subplots(2, len(slips) + 1, figsize=(3.4 * (len(slips) + 1), 6.4),
                             sharex=True, sharey=True)
    for r, (key, ylabel) in enumerate(ROWS):
        for c, slip in enumerate(slips):
            ax = axes[r, c]
            group = [t for t in trials if t['slip'] == slip]
            for t in group:
                w = (t['t'] >= WINDOW[0]) & (t['t'] <= WINDOW[1])
                ax.plot(t['t'][w], t[key][w], color=colors[slip], lw=0.8, alpha=0.35)
            mu, sd = mean_sd(group, key, grid)
            ax.plot(grid, mu, color=colors[slip], lw=2.2)
            ax.fill_between(grid, mu - sd, mu + sd, color=colors[slip], alpha=0.15, lw=0)
            axes[r, -1].plot(grid, mu, color=colors[slip], lw=2.2)
            if r == 0:
                ax.set_title(f'Slip {slip}  (n={len(group)})', loc='left', fontsize=10)
        axes[r, 0].set_ylabel(ylabel)
    axes[0, -1].set_title('Slip means', loc='left', fontsize=10)
    axes[0, -1].legend(handles=[Line2D([], [], color=colors[s], lw=2.2, label=f'Slip {s}')
                                for s in slips], loc='lower right', frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.axvline(0, color='#888888', lw=1, ls=':')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_xlim(*WINDOW)
    for ax in axes[-1]:
        ax.set_xlabel('Time from R foot gyro onset (s)')
    fig.suptitle('Itak: right exo torque per slip level (torque-on trials)\n'
                 'thin = trials, bold = mean, band = ±1 SD; − = plantarflexion',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)
    out = os.path.join(BASE, 'figures', 'itak_torque_per_slip.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')

    rows = []
    for t in trials:
        w = (t['t'] >= WINDOW[0]) & (t['t'] <= WINDOW[1])
        rows.append(dict(slip=t['slip'], trial=t['trial'],
                         peak_sent=t['sent'][w].min(), peak_measured=t['measured'][w].min(),
                         t_peak_measured=t['t'][w][np.argmin(t['measured'][w])]))
    print(pd.DataFrame(rows).groupby('slip')[['peak_sent', 'peak_measured', 't_peak_measured']]
          .agg(['mean', 'std']).to_string(float_format=lambda x: f'{x:6.2f}'))


if __name__ == '__main__':
    main()
