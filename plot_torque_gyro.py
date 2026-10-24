"""Per-slip plots of right exo measured torque and right foot gyro Z for the
exo-torque-on trials, aligned to perturbation (belt) onset.

Alignment and onset detection are the same as plot_aligned_ankle_exo.py
(Vicon trigger <-> exo gpio edge; tread1 marker onset, falling back to
sim_test + mean offset), but no IK/ID files are required.

Torque = red, gyro = blue; successful trials full opacity, unsuccessful light
(Success column of Trial_Info.csv).
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_aligned_ankle_exo import read_vicon_csv, tread1_onset, vicon_trigger_time

BASE = os.path.dirname(os.path.abspath(__file__))
WINDOW = (-0.4, 1.6)  # s relative to belt onset
TORQUE_COLOR = '#d62728'
GYRO_COLOR = '#1f5fd6'
FAILED_ALPHA = 0.3  # unsuccessful trials drawn light


def load_torque_on_trials():
    info = pd.read_csv(os.path.join(BASE, 'Trial_Info.csv'))
    info.columns = [c.strip() for c in info.columns]
    info = info.set_index('Trial Number')

    trials = []
    for n in info.index:
        slip = int(info.loc[n, 'Slip Level'])
        if str(info.loc[n, 'Torque']).strip() != 'y':
            continue
        name = f'Trial_{n}_slip{slip}'
        marker = os.path.join(BASE, 'Marker', name + '.csv')
        exo = glob.glob(os.path.join(BASE, 'IMU_Torque', f'Pert_Policy_trial_{n}_slip_{slip}_*_data.csv'))
        sim = os.path.join(BASE, 'sim_test', name + '.txt')
        if not os.path.exists(marker) or len(exo) != 1:
            print(f'  Trial {n} (slip {slip}): missing marker or exo csv, skipped')
            continue
        devices, names, traj = read_vicon_csv(marker)
        t_vtrig = vicon_trigger_time(devices)
        d = pd.read_csv(exo[0])
        gpio_high = d['gpio_output'].values > 0.5
        if t_vtrig is None or not gpio_high.any():
            print(f'  Trial {n} (slip {slip}): no trigger edge, skipped')
            continue
        trials.append(dict(
            n=n, slip=slip, success=int(info.loc[n, 'Success']) == 1, exo=d,
            t_vtrig=t_vtrig, t_gpio=d['timestamp'].values[np.argmax(gpio_high)],
            tread_onset=tread1_onset(names, traj),
            sim_onset=float(open(sim).read()) if os.path.exists(sim) else None))

    # sim_test -> tread1 offset, as in plot_aligned_ankle_exo.py
    diffs = [t['tread_onset'] - t['sim_onset'] for t in trials
             if t['tread_onset'] is not None and t['sim_onset'] is not None]
    offset = float(np.mean(diffs))

    kept = []
    for t in trials:
        if t['tread_onset'] is not None:
            onset = t['tread_onset']
        elif t['sim_onset'] is not None:
            onset = t['sim_onset'] + offset
        else:
            print(f"  Trial {t['n']} (slip {t['slip']}): no onset, skipped")
            continue
        t['t'] = t['exo']['timestamp'].values - t['t_gpio'] + t['t_vtrig'] - onset
        kept.append(t)
    return kept


def main():
    trials = load_torque_on_trials()
    rows = [('torque_measured_R', 'Right measured torque (N·m)', TORQUE_COLOR),
            ('foot_right_gyro_z', 'Right foot gyro Z (rad/s)', GYRO_COLOR)]

    figs = []
    for slip in sorted({t['slip'] for t in trials}):
        group = sorted((t for t in trials if t['slip'] == slip), key=lambda t: t['n'])
        n_ok = sum(t['success'] for t in group)
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
        for ax, (col, ylabel, color) in zip(axes, rows):
            for t in group:
                w = (t['t'] >= WINDOW[0]) & (t['t'] <= WINDOW[1])
                ax.plot(t['t'][w], t['exo'][col].values[w], color=color, lw=1.4,
                        alpha=1.0 if t['success'] else FAILED_ALPHA)
            ax.axvline(0, color='#444444', lw=1, ls='--')
            ax.axhline(0, color='#cccccc', lw=0.8)
            ax.set_ylabel(ylabel)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        axes[0].text(0.01, 0.04, 'belt onset', transform=axes[0].transAxes, fontsize=8, color='#444444')
        axes[1].set_xlabel('Time relative to belt onset (s)')
        axes[1].set_xlim(*WINDOW)
        axes[0].set_title(f'Slip {slip}   torque on   n = {len(group)}', loc='left')

        # legend above the data (outside the axes) so it never covers a trace
        handles = [Line2D([], [], color='#555555', lw=1.4, label=f'Successful (n={n_ok})')]
        if n_ok < len(group):
            handles.append(Line2D([], [], color='#555555', lw=1.4, alpha=FAILED_ALPHA,
                                  label=f'Unsuccessful (n={len(group) - n_ok})'))
        axes[0].legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.0),
                       ncol=len(handles), frameon=False, fontsize=8, borderaxespad=0.2)
        figs.append((slip, fig, axes))

    # same y-limits across slips
    for r in range(len(rows)):
        lo = min(a[r].get_ylim()[0] for _, _, a in figs)
        hi = max(a[r].get_ylim()[1] for _, _, a in figs)
        for _, _, a in figs:
            a[r].set_ylim(lo, hi)

    for slip, fig, _ in figs:
        fig.tight_layout()
        out = os.path.join(BASE, 'figures', f'slip_{slip}_torque_gyro.png')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
