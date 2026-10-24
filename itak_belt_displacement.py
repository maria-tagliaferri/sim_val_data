"""Itak: belt displacement per trial from the tread1 marker, summarised per
slip level (7, 8, 9).

Displacement = |tread1 X at the end of the trial − median of the first 50
frames| (the belt moves once and then holds). Peak speed is from the
differentiated X position. Duration runs from onset (> 1 mm, as in
tread1_onset) to the first frame within 1 mm of the final position.
Trial_1_* files are practice/familiarisation and are left out.

Writes figures/itak_belt_displacement.csv.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from plot_aligned_ankle_exo import TREAD_THRESHOLD_MM, read_vicon_csv

BASE = os.path.dirname(os.path.abspath(__file__))
ITAK = os.path.join(BASE, 'Itak')
SLIPS = (7, 8, 9)
NAME = re.compile(r'Trial_(\d+)_slip(\d+)\.csv')


def belt_metrics(path):
    _, names, traj = read_vicon_csv(path)
    cols = [i for i, s in enumerate(names) if s.endswith(':tread1')]
    if not cols:
        return None
    x = traj[:, cols[0]]
    t = (traj[:, 0] - 1) / 100
    ok = ~np.isnan(x)
    x0, x_end = np.median(x[ok][:50]), np.median(x[ok][-50:])
    dev = np.abs(x - x0)
    i_on = np.argmax(ok & (dev > TREAD_THRESHOLD_MM))
    i_off = i_on + np.argmax(np.abs(x[i_on:] - x_end) < TREAD_THRESHOLD_MM)
    speed = np.abs(np.gradient(x[ok], t[ok]))
    return dict(displacement_mm=abs(x_end - x0), peak_speed_mm_s=speed.max(),
                duration_s=t[i_off] - t[i_on])


def main():
    info = pd.read_csv(os.path.join(ITAK, 'Itak_trial_info.csv'), dtype=str)
    info.columns = [c.strip() for c in info.columns]
    info = info[info['Trial Number'].fillna('').str.strip().str.isdigit()]
    torque = {int(n): str(v).strip() for n, v in zip(info['Trial Number'], info['Torque'])}

    rows = []
    for path in glob.glob(os.path.join(ITAK, 'Marker', 'Trial_*_slip*.csv')):
        n, slip = map(int, NAME.search(os.path.basename(path)).groups())
        if slip not in SLIPS or n == 1:
            continue
        m = belt_metrics(path)
        if m is None:
            print(f'Trial {n} (slip {slip}): no tread1 marker, skipped')
            continue
        rows.append(dict(trial=n, slip=slip, torque=torque.get(n, '?'), **m))
    df = pd.DataFrame(rows).sort_values(['slip', 'trial'])
    df.to_csv(os.path.join(BASE, 'figures', 'itak_belt_displacement.csv'), index=False)

    fmt = lambda x: f'{x:8.1f}'
    print(df.to_string(index=False, float_format=fmt))
    print('\nPer slip level (mean, SD):')
    print(df.groupby('slip')[['displacement_mm', 'peak_speed_mm_s', 'duration_s']]
          .agg(['count', 'mean', 'std']).to_string(float_format=lambda x: f'{x:7.2f}'))
    print('\nPer slip level and torque condition (displacement, mm):')
    print(df.groupby(['slip', 'torque'])['displacement_mm'].agg(['count', 'mean', 'std'])
          .to_string(float_format=lambda x: f'{x:7.2f}'))


if __name__ == '__main__':
    main()
