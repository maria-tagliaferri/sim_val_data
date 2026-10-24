"""Rajiv_Val: belt displacement per perturbation level from the tread1 marker,
and the threshold shift expressed in belt displacement.

Displacement / peak speed / duration per trial use belt_metrics() from
itak_belt_displacement.py. Belt displacement at each staircase threshold
(outcomes.txt: no torque 10.08, torque 11.17) is interpolated from a linear
fit of displacement vs level over all trials.

Trial 9 is excluded: tread1 reverses after ~6 cm and is then lost for the
rest of the trial, so its final belt position is not measured.

Writes figures/rajiv_val/rajiv_val_belt_displacement.csv.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from itak_belt_displacement import belt_metrics

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'Rajiv_Val')
THRESHOLDS = {'no torque': 10.08, 'torque': 11.17}  # from Rajiv_Val/outcomes.txt
EXCLUDE = {9}  # tread1 lost mid-perturbation


def main():
    info = pd.read_csv(os.path.join(DATA, 'Simulation data collection - Rajiv_Val.csv'))
    info.columns = [c.strip() for c in info.columns]
    info = info.set_index('Trial Number')

    rows = []
    for path in glob.glob(os.path.join(DATA, 'Markers', 'Trial_*.csv')):
        n = int(re.search(r'Trial_(\d+)_', os.path.basename(path)).group(1))
        if n in EXCLUDE:
            continue
        m = belt_metrics(path)
        if m is None:
            print(f'Trial {n}: no tread1 marker, skipped')
            continue
        rows.append(dict(trial=n, level=float(info.loc[n, 'Slip Level']),
                         torque=str(info.loc[n, 'Torque']).strip(), **m))
    df = pd.DataFrame(rows).sort_values(['level', 'trial'])
    df['displacement_cm'] = df['displacement_mm'] / 10
    df['peak_speed_cm_s'] = df['peak_speed_mm_s'] / 10
    os.makedirs(os.path.join(BASE, 'figures', 'rajiv_val'), exist_ok=True)
    df.to_csv(os.path.join(BASE, 'figures', 'rajiv_val', 'rajiv_val_belt_displacement.csv'), index=False)

    print(df[['trial', 'level', 'torque', 'displacement_cm', 'peak_speed_cm_s', 'duration_s']]
          .to_string(index=False, float_format=lambda x: f'{x:6.2f}'))
    print('\nPer level (mean, SD):')
    print(df.groupby('level')[['displacement_cm', 'peak_speed_cm_s', 'duration_s']]
          .agg(['count', 'mean', 'std']).to_string(float_format=lambda x: f'{x:6.2f}'))

    slope, icpt = np.polyfit(df['level'], df['displacement_cm'], 1)
    r = np.corrcoef(df['level'], df['displacement_cm'])[0, 1]
    print(f'\nLinear fit: displacement = {slope:.2f} cm/level × level + {icpt:.2f} (r = {r:.3f})')
    at = {k: slope * v + icpt for k, v in THRESHOLDS.items()}
    for k, v in at.items():
        print(f'  {k:9s} threshold level {THRESHOLDS[k]:.2f} -> {v:.2f} cm')
    d = at['torque'] - at['no torque']
    print(f'  threshold shift: +{d:.2f} cm (+{100 * d / at["no torque"]:.1f} %)')

    means = df.groupby('level')['displacement_cm'].mean()
    d10 = means[11.0] - means[10.0]
    print(f'\nLevel 10 -> level 11 (measured means): {means[10.0]:.2f} -> {means[11.0]:.2f} cm, '
          f'+{d10:.2f} cm (+{100 * d10 / means[10.0]:.1f} %)')


if __name__ == '__main__':
    main()
