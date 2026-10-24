"""Compare commanded ankle torque (torque actually sent to the motors) across
subjects.

Subjects are the three exo-log folders:
    IMU_Torque (Maria), Rafay, Rajiv, Itak/IMU (Itak)

Only torque-on trials are plotted (filename does not contain 'notorque').
Rafay is limited to the successful trials: 6, 13, 17, and 25.
torque_sent_R is the command written to the right motor. torque_sent_L is
exactly -torque_sent_R on every sample, so the left motor is the mirror image
and is not plotted separately.

Time is aligned to right-foot gyro onset: the first sample where
|foot_right_gyro_z| exceeds 0.5 rad/s above the median of the first 0.5 s.
On the IMU_Torque recordings this onset sits about 0.13 s after belt motion
(see figures/alignment_onsets.csv), and the same rule is used for every
subject so the comparison does not depend on Vicon.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
# Folder name -> label on the figures. IMU_Torque is Maria's session.
FOLDERS = (('IMU_Torque', 'Maria'), ('Rafay', 'Rafay'), ('Rajiv', 'Rajiv'),
           (os.path.join('Itak', 'IMU'), 'Itak'))
SUBJECTS = tuple(label for _, label in FOLDERS)
COLORS = {'Maria': '#0072B2', 'Rafay': '#D55E00', 'Rajiv': '#009E73', 'Itak': '#CC79A7'}
WINDOW = (-0.35, 1.0)  # s relative to right-foot gyro onset
GYRO_THR = 0.5  # rad/s above the quiet-standing median
DT = 0.025
NAME = re.compile(r'Pert_Policy_trial_(\d+)_slip_(\d+)(_notorque)?_')
# Successful Rafay trials only. Other subjects keep every torque-on trial.
KEEP = {'Rafay': {6, 13, 17, 25}}


def load_trials():
    trials = []
    for folder, subject in FOLDERS:
        for path in sorted(glob.glob(os.path.join(BASE, folder, 'Pert_Policy_trial_*_data.csv'))):
            m = NAME.search(os.path.basename(path))
            if m is None or m.group(3):
                continue
            trial = int(m.group(1))
            if subject in KEEP and trial not in KEEP[subject]:
                continue
            d = pd.read_csv(path)
            ts = d['timestamp'].values
            gyro = d['foot_right_gyro_z'].values
            n_base = max(8, int(round(0.5 / np.median(np.diff(ts)))))
            baseline = np.median(gyro[:n_base])
            moved = np.where(np.abs(gyro - baseline) > GYRO_THR)[0]
            if len(moved) == 0:
                print(f'  {subject} trial {m.group(1)} slip {m.group(2)}: no gyro onset, skipped')
                continue
            trials.append(dict(
                subject=subject, trial=trial, slip=int(m.group(2)),
                t=ts - ts[moved[0]], sent=d['torque_sent_R'].values))
    return trials


def mean_trace(group, grid):
    stack = np.vstack([np.interp(grid, t['t'], t['sent'], left=np.nan, right=np.nan) for t in group])
    count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    mu = np.full(grid.shape, np.nan)
    sd = np.full(grid.shape, np.nan)
    ok = count > 0
    mu[ok] = total[ok] / count[ok]
    if np.any(count > 1):
        var = np.nansum((stack - mu) ** 2, axis=0)
        sd[count > 1] = np.sqrt(var[count > 1] / (count[count > 1] - 1))
    return mu, sd


def draw(ax, trials, grid, show_mean=True):
    for subject in SUBJECTS:
        group = [t for t in trials if t['subject'] == subject]
        for t in group:
            w = (t['t'] >= WINDOW[0]) & (t['t'] <= WINDOW[1])
            ax.plot(t['t'][w], t['sent'][w], color=COLORS[subject], lw=1.0, alpha=0.35)
        if show_mean and group:
            mu, sd = mean_trace(group, grid)
            ax.plot(grid, mu, color=COLORS[subject], lw=2.2)
            ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[subject], alpha=0.15, lw=0)
    ax.axhline(0, color='#cccccc', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*WINDOW)


def legend_handles(trials, with_n=True):
    handles = []
    for subject in SUBJECTS:
        n = sum(t['subject'] == subject for t in trials)
        if n == 0:
            continue
        label = f'{subject} (n={n})' if with_n else subject
        handles.append(Line2D([], [], color=COLORS[subject], lw=2.2, label=label))
    return handles


def main():
    trials = load_trials()
    slips = sorted({t['slip'] for t in trials})
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)

    # Shared y-limits from every torque-on trial inside the window.
    samples = np.concatenate([
        t['sent'][(t['t'] >= WINDOW[0]) & (t['t'] <= WINDOW[1])] for t in trials])
    pad = 0.08 * (samples.max() - samples.min())
    ylim = (samples.min() - pad, max(2.0, samples.max() + pad))

    figs = []
    for slip in slips:
        group = [t for t in trials if t['slip'] == slip]
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        draw(ax, group, grid)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Right commanded torque (N·m)')
        ax.set_title(f'Slip {slip}    torque sent to the motors', loc='left')
        ax.legend(handles=legend_handles(group), loc='lower right', frameon=False, fontsize=8)
        fig.tight_layout()
        out = os.path.join(BASE, 'figures', f'commanded_torque_slip_{slip}.png')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        figs.append(out)
        print(f'Saved {out}')

    # One figure with every slip, so subjects can be compared at a glance.
    fig, axes = plt.subplots(1, len(slips), figsize=(3.3 * len(slips), 4.0), sharey=True)
    if len(slips) == 1:
        axes = [axes]
    for ax, slip in zip(axes, slips):
        group = [t for t in trials if t['slip'] == slip]
        draw(ax, group, grid)
        counts = '   '.join(
            f'{s} n={sum(t["subject"] == s for t in group)}'
            for s in SUBJECTS if any(t['subject'] == s for t in group))
        ax.set_title(f'Slip {slip}\n{counts}', loc='left', fontsize=9)
    axes[0].set_ylabel('Right commanded torque (N·m)')
    for ax in axes:
        ax.set_ylim(*ylim)
        ax.set_xlabel('Time (s)')
    fig.legend(handles=legend_handles(trials, with_n=False), loc='upper center', ncol=len(SUBJECTS),
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Commanded right-ankle torque sent to the motors', y=1.08, fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE, 'figures', 'commanded_torque_subjects.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # Peak commanded torque in the first 0.6 s after gyro onset.
    rows = []
    for t in trials:
        w = (t['t'] >= WINDOW[0]) & (t['t'] <= WINDOW[1])
        i = np.argmax(np.abs(t['sent'][w]))
        rows.append(dict(subject=t['subject'], trial=t['trial'], slip=t['slip'],
                         peak_Nm=t['sent'][w][i]))
    peaks = pd.DataFrame(rows).sort_values(['slip', 'subject', 'trial'])
    peak_path = os.path.join(BASE, 'figures', 'commanded_torque_peaks.csv')
    peaks.to_csv(peak_path, index=False)
    print(peaks.groupby(['slip', 'subject'])['peak_Nm'].agg(['count', 'mean', 'min', 'max'])
          .to_string(float_format=lambda x: f'{x:7.2f}'))
    print(f'Saved {peak_path}')


if __name__ == '__main__':
    main()
