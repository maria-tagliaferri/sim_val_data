"""Belt translation measured by the tread1 marker, mean +/- SD per slip level,
to check how repeatable the perturbation is.

Every marker trial from every subject at slip 7-11.5 is used (torque and
no-torque, successful or not), including Rajiv's half levels (9.5, 10.5, 11.5).
The slip level is taken from the marker filename (Trial_10_slip9.csv,
Trial_10_slipslip11p0.csv); it matches Trial_Info.csv except for a few
repeated / unlisted trials where the filename is the reliable one.

Translation = -(tread1 X - starting X), in cm, so the belt moving backwards is
positive. Time 0 = tread1 onset (first frame > 1 mm from the start position).

Trials where tread1 comes forward again by more than REVERSAL_CM after its
peak are dropped: the marker was placed close enough to the end of the
treadmill that it went over the roller, so it stops following the belt.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import pandas as pd

from plot_aligned_ankle_exo import read_vicon_csv, tread1_onset

BASE = os.path.dirname(os.path.abspath(__file__))
# label -> (subject folder, marker folder)
SUBJECTS = {
    'Maria': ('Maria_Val', 'Marker'),
    'Rafay': ('Rafay_Val', 'Marker'),
    'Rajiv': ('Rajiv_Val', 'Markers'),
    'Itak': ('Itak_Val', 'Marker'),
}
COLORS = {'Maria': '#0072B2', 'Rafay': '#D55E00', 'Rajiv': '#009E73', 'Itak': '#CC79A7'}
LEVELS = (7, 8, 9, 9.5, 10, 10.5, 11, 11.5)
NAME = re.compile(r'Trial_(\d+)_slip(?:slip)?(\d+(?:p\d+)?)\.csv$')
WINDOW = (-0.2, 1.0)  # s relative to tread1 onset
FINAL = (0.8, 1.0)  # s after onset averaged for the final translation (belt has stopped)
DT = 0.01
REVERSAL_CM = 1.0  # drop from peak translation that flags a trial as reversed


def load_trials():
    trials = []
    for subject, (root, marker_dir) in SUBJECTS.items():
        for path in sorted(glob.glob(os.path.join(BASE, root, marker_dir, 'Trial_*.csv'))):
            m = NAME.search(os.path.basename(path))
            if m is None:
                continue
            n, slip = int(m.group(1)), float(m.group(2).replace('p', '.'))
            if slip not in LEVELS:
                continue
            _, names, traj = read_vicon_csv(path)
            onset = tread1_onset(names, traj)
            if onset is None:
                print(f'  {subject} trial {n} (slip {slip:g}): no usable tread1, skipped')
                continue
            x = traj[:, [i for i, s in enumerate(names) if s.endswith(':tread1')][0]]
            t = (traj[:, 0] - 1) / 100 - onset
            disp = -(x - np.nanmedian(x[:50])) / 10  # mm -> cm, backwards = positive
            w = (t >= 0) & (t <= WINDOW[1]) & np.isfinite(disp)
            if disp[w].max() - disp[w][-1] > REVERSAL_CM:
                print(f'  {subject} trial {n} (slip {slip:g}): tread1 went over the roller, skipped')
                continue
            trials.append(dict(subject=subject, trial=n, slip=slip, t=t, disp=disp,
                               vel=np.gradient(disp, t)))
    return trials


def main():
    trials = load_trials()
    levels = [lv for lv in LEVELS if any(t['slip'] == lv for t in trials)]
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)
    # Viridis, yellow (low level) -> dark purple (high level): ordered but each level a distinct hue
    colors = {lv: cm.viridis(0.9 * (1 - i / (len(levels) - 1))) for i, lv in enumerate(levels)}

    rows = []
    for t in trials:
        final = np.nanmean(t['disp'][(t['t'] >= FINAL[0]) & (t['t'] <= FINAL[1])])
        w = (t['t'] >= 0) & (t['t'] <= FINAL[0])
        rows.append(dict(subject=t['subject'], trial=t['trial'], slip=t['slip'],
                         final_cm=final, peak_vel_cm_s=np.nanmax(t['vel'][w])))
    per_trial = pd.DataFrame(rows).sort_values(['slip', 'subject', 'trial'])

    # Time courses: every level on one axes
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ends = []
    for slip in levels:
        group = [t for t in trials if t['slip'] == slip]
        stack = np.vstack([np.interp(grid, t['t'], t['disp'], left=np.nan, right=np.nan) for t in group])
        mu = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1) if len(group) > 1 else np.zeros(grid.shape)
        ax.fill_between(grid, mu - sd, mu + sd, color=colors[slip], alpha=0.2, lw=0)
        ax.plot(grid, mu, color=colors[slip], lw=2, ls='--' if slip % 1 else '-',  # half levels dashed
                label=f'Slip {slip:g} (n={len(group)})')
        ends.append([mu[-1], slip])
    # Direct labels at the right end, nudged apart so neighbouring levels don't overlap
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 1.3)
    for y, slip in ends:
        ax.text(WINDOW[1] + 0.02, y, f'{slip:g}', color='#333333', fontsize=8, va='center')
    ax.axvline(0, color='#bbbbbb', lw=0.8)
    ax.set_xlim(*WINDOW)
    ax.set_xlabel('Time from tread1 onset (s)')
    ax.set_ylabel('Belt translation (cm)')
    ax.set_title('Belt translation from the tread1 marker, mean ± SD across subjects', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(BASE, 'figures', 'belt_translation_by_slip.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')

    # Summary: final translation and peak velocity per level, each trial a dot coloured by subject
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    rng = np.random.default_rng(0)
    stats = per_trial.groupby('slip')[['final_cm', 'peak_vel_cm_s']].agg(['mean', 'std'])
    for ax, col, label in zip(axes, ('final_cm', 'peak_vel_cm_s'),
                              ('Final belt translation (cm)', 'Peak belt velocity (cm/s)')):
        for subject in SUBJECTS:
            sub = per_trial[per_trial['subject'] == subject]
            ax.scatter(sub['slip'] + rng.uniform(-0.1, 0.1, len(sub)), sub[col], s=14,
                       color=COLORS[subject], alpha=0.75, lw=0, label=subject)
        ax.errorbar(stats.index, stats[col]['mean'], yerr=stats[col]['std'].fillna(0), fmt='_',
                    color='#222222', ms=14, mew=2, capsize=4, lw=1.2, label='Mean ± SD')
        ax.set_xticks(levels)
        ax.set_xticklabels([f'{lv:g}' for lv in levels], fontsize=8)
        ax.set_xlabel('Slip level')
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    out = os.path.join(BASE, 'figures', 'belt_translation_summary.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')

    summary = per_trial.groupby('slip').agg(
        n=('final_cm', 'size'), final_mean_cm=('final_cm', 'mean'), final_sd_cm=('final_cm', 'std'),
        peak_vel_mean=('peak_vel_cm_s', 'mean'), peak_vel_sd=('peak_vel_cm_s', 'std'))
    summary['final_cv_pct'] = 100 * summary['final_sd_cm'] / summary['final_mean_cm']
    print(summary.to_string(float_format=lambda v: f'{v:7.2f}'))
    out = os.path.join(BASE, 'figures', 'belt_translation_per_trial.csv')
    per_trial.to_csv(out, index=False, float_format='%.3f')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
