"""Torque-on minus no-torque difference in peak AP COM state (COM - feet centre),
successful trials only, per subject.

Reads the per-trial time series written by plot_com_state.py
(figures/com_state/com_state_<subject>.csv). Peaks are taken in the first
PEAK_WINDOW s after perturbation onset:
    peak displacement   max of disp_cm
    peak velocity       max of vel_cm_s
    peak acceleration   max of acc_m_s2 (forward, from the belt pulling the feet back)
    peak deceleration   min of acc_m_s2 (braking of the forward COM motion)

Only slip levels where the subject has successful trials in both conditions
are used. Each torque-on trial is compared with the mean of the no-torque
trials at the same level (difference = torque - no torque), and those
differences are averaged per subject (mean +/- SD). The per-level table gives
both condition means and the difference of the means.
"""
import glob
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PEAK_WINDOW = (0.0, 1.0)  # s after perturbation onset
METRICS = {
    'peak_disp_cm': ('disp_cm', 'max'),
    'peak_vel_cm_s': ('vel_cm_s', 'max'),
    'peak_acc_m_s2': ('acc_m_s2', 'max'),
    'peak_decel_m_s2': ('acc_m_s2', 'min'),
}


def main():
    d = pd.concat(pd.read_csv(p) for p in glob.glob(os.path.join(BASE, 'figures', 'com_state', 'com_state_*.csv')))
    d = d[(d['success'] == 1) & d['time_from_pert_s'].between(*PEAK_WINDOW)]
    keys = ['subject', 'slip', 'torque', 'trial']
    peaks = d.groupby(keys).agg(**{m: (col, how) for m, (col, how) in METRICS.items()}).reset_index()

    # Keep levels that have both conditions for that subject
    both = peaks.groupby(['subject', 'slip'])['torque'].transform('nunique') == 2
    peaks = peaks[both]

    # Each torque-on trial minus the no-torque mean at the same subject / level
    nt_mean = peaks[peaks['torque'] == 'n'].groupby(['subject', 'slip'])[list(METRICS)].mean()
    on = peaks[peaks['torque'] == 'y'].set_index(['subject', 'slip'])
    diffs = (on[list(METRICS)] - nt_mean.reindex(on.index)).reset_index()

    per_subject = diffs.groupby('subject')[list(METRICS)].agg(['mean', 'std'])
    per_subject.insert(0, ('levels', ''), diffs.groupby('subject')['slip'].apply(
        lambda s: ', '.join(f'{v:g}' for v in sorted(s.unique()))))
    per_subject.insert(1, ('n_torque', ''), diffs.groupby('subject').size())
    per_subject.insert(2, ('n_no_torque', ''), peaks[peaks['torque'] == 'n'].groupby('subject').size())

    per_level = peaks.groupby(['subject', 'slip', 'torque'])[list(METRICS)].agg(['mean', 'std', 'size'])

    pd.set_option('display.width', 200)
    fmt = lambda v: f'{v:7.2f}'
    print('Torque - no torque, per subject (mean and SD over torque-on trials):')
    print(per_subject.to_string(float_format=fmt))
    print('\nPer level and condition:')
    print(per_level.to_string(float_format=fmt))

    out_dir = os.path.join(BASE, 'figures', 'com_state')
    per_subject.to_csv(os.path.join(out_dir, 'com_peak_diff_per_subject.csv'), float_format='%.3f')
    per_level.to_csv(os.path.join(out_dir, 'com_peak_per_level.csv'), float_format='%.3f')
    peaks.to_csv(os.path.join(out_dir, 'com_peaks_per_trial.csv'), index=False, float_format='%.3f')
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
