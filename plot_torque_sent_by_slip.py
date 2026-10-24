"""Mean +/- SD of the commanded exo torque (torque_sent_R) at each slip level,
with every subject on the same axes.

Trials
------
All torque-on trials (Trial_Info.csv: Torque == 'y'), successful or not, at
slip levels 8, 9, 10, 11 for Maria, Rajiv and Itak. Rajiv's half levels
(10.5, 11.5) are not included. Rafay is left out for now because the exo
torque malfunctioned on many of his trials.

Time base
---------
- Perturbation onset (Vicon time): first frame where the tread1 marker X moves
  > 1 mm from its starting position. Trials with no tread1 data (Maria 8-13)
  are skipped.
- Exo -> Vicon sync: exo gpio_output rising edge = Vicon trigger rising edge.
  Trials with no Vicon trigger pulse (all of Itak's) are placed using the
  right-foot gyro onset minus the mean gyro latency after perturbation onset,
  measured on that subject's triggered trials, or on every subject's triggered
  trials when the subject has none.
- Ankle response onset (Vicon time): first sample after perturbation onset
  where ID ankle_angle_r_moment falls below the pre-perturbation baseline by
  max(3 SD, 20% of that trial's peak drop) and stays there for 50 ms. The hold
  time skips the 1-2 sample force-plate transient at belt onset. The ID model has no exo actuator,
  so this is the net moment (biological + exo); set SUBTRACT_EXO = True to
  detect onset on ID - torque_measured_R instead.

All trials are aligned so t = 0 is the ankle response onset. The dashed lines
show each subject's mean perturbation onset on that time base.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_aligned_ankle_exo import read_sto, read_vicon_csv, tread1_onset, vicon_trigger_time

BASE = os.path.dirname(os.path.abspath(__file__))
# label -> (subject folder, exo log folder, marker folder)
SUBJECTS = {
    'Maria': ('Maria_Val', 'IMU_Torque', 'Marker'),
    # Rafay ('Rafay_Val', 'IMU', 'Marker') left out: exo torque malfunctioned on many trials
    'Rajiv': ('Rajiv_Val', 'IMU', 'Markers'),
    'Itak': ('Itak_Val', 'IMU', 'Marker'),  # no Vicon trigger, synced by gyro onset
}
COLORS = {'Maria': '#0072B2', 'Rafay': '#D55E00', 'Rajiv': '#009E73', 'Itak': '#CC79A7'}
LEVELS = (8, 9, 10, 11)
WINDOW = (-0.3, 1.0)  # s relative to ankle response onset
DT = 0.01
BASELINE = 0.5  # s of quiet standing before perturbation used for the ID baseline
ID_SD_K, ID_PEAK_FRAC, ID_HOLD = 3.0, 0.2, 5  # onset threshold and samples it must hold
GYRO_THR = 0.5  # rad/s, right-foot gyro onset for trials without a Vicon trigger
SUBTRACT_EXO = False


def load_info(root):
    # Trial_Info.csv, or Itak_trial_info.csv for Itak
    path = [p for p in glob.glob(os.path.join(BASE, root, '*.csv')) if p.lower().endswith('trial_info.csv')][0]
    info = pd.read_csv(path)
    info.columns = [c.strip() for c in info.columns]
    info['Trial Number'] = pd.to_numeric(info['Trial Number'], errors='coerce')
    info['Slip Level'] = pd.to_numeric(info['Slip Level'], errors='coerce')
    info['Success'] = pd.to_numeric(info['Success'], errors='coerce')
    info['Torque'] = info['Torque'].astype(str).str.strip()
    return info[(info['Torque'] == 'y') & info['Slip Level'].isin(LEVELS)]


def gyro_onset(d, after):
    t = d['timestamp'].values
    g = d['foot_right_gyro_z'].values
    base = np.median(g[t < after])
    moved = np.where((np.abs(g - base) > GYRO_THR) & (t > after))[0]
    return t[moved[0]] if len(moved) else None


def response_onset(t, m, pert):
    pre = (t >= pert - BASELINE) & (t < pert)
    mu, sd = m[pre].mean(), m[pre].std()
    drop = mu - m[(t >= pert) & (t < pert + 0.5)].min()
    below = (m < mu - max(ID_SD_K * sd, ID_PEAK_FRAC * drop)) & (t >= pert)
    held = np.convolve(below, np.ones(ID_HOLD, dtype=int), 'valid') == ID_HOLD
    idx = np.where(held)[0]
    return t[idx[0]] if len(idx) else None


def load_trials():
    trials = []
    for subject, (root, exo_dir, marker_dir) in SUBJECTS.items():
        for _, row in load_info(root).iterrows():
            n, slip = int(row['Trial Number']), int(row['Slip Level'])
            marker = glob.glob(os.path.join(BASE, root, marker_dir, f'Trial_{n}_slip*.csv'))
            id_file = glob.glob(os.path.join(BASE, root, 'ID', f'inverse_dynamics_Trial_{n}_slip*.sto'))
            exo = [p for p in glob.glob(os.path.join(BASE, root, exo_dir, f'Pert_Policy_trial_{n}_slip_*_data.csv'))
                   if 'notorque' not in p]
            if not (marker and id_file and len(exo) == 1):
                print(f'  {subject} trial {n} (slip {slip}): missing marker / ID / exo file, skipped')
                continue
            devices, names, traj = read_vicon_csv(marker[0])
            d = pd.read_csv(exo[0])
            t_trig = vicon_trigger_time(devices)
            gpio = d['gpio_output'].values > 0.5
            trials.append(dict(
                subject=subject, trial=n, slip=slip, success=row['Success'] == 1, exo=d, id=read_sto(id_file[0]),
                pert=tread1_onset(names, traj),
                # exo timestamp + offset = Vicon time (None until resolved below)
                offset=t_trig - d['timestamp'].values[np.argmax(gpio)] if t_trig is not None and gpio.any() else None))

    for t in trials:
        if t['pert'] is None:
            print(f'  {t["subject"]} trial {t["trial"]} (slip {t["slip"]}): no tread1 onset, skipped')
    trials = [t for t in trials if t['pert'] is not None]

    # Gyro latency from the synced trials, used to place trials that have no trigger
    lat = {}
    for t in trials:
        if t['offset'] is not None:
            g = gyro_onset(t['exo'], t['pert'] - t['offset'] - 0.5)
            if g is not None:
                lat.setdefault(t['subject'], []).append(g + t['offset'] - t['pert'])
    pooled = np.mean([x for v in lat.values() for x in v])
    kept = []
    for t in trials:
        t['sync'] = 'trigger'
        if t['offset'] is None:
            g = gyro_onset(t['exo'], 1.0)
            if g is None:
                print(f'  {t["subject"]} trial {t["trial"]}: no trigger and no gyro onset, skipped')
                continue
            t['offset'] = t['pert'] + np.mean(lat.get(t['subject'], [pooled])) - g
            t['sync'] = 'gyro'

        exo_t = t['exo']['timestamp'].values + t['offset']
        id_t = t['id']['time'].values
        moment = t['id']['ankle_angle_r_moment'].values
        if SUBTRACT_EXO:
            moment = moment - np.interp(id_t, exo_t, t['exo']['torque_measured_R'].values)
        resp = response_onset(id_t, moment, t['pert'])
        if resp is None:
            print(f'  {t["subject"]} trial {t["trial"]}: no ID ankle response found, skipped')
            continue
        t.update(resp=resp, t=exo_t - resp, sent=t['exo']['torque_sent_R'].values)
        kept.append(t)
    return kept


def mean_sd(group, grid):
    stack = np.vstack([np.interp(grid, t['t'], t['sent'], left=np.nan, right=np.nan) for t in group])
    sd = np.nanstd(stack, axis=0, ddof=1) if len(group) > 1 else np.zeros(grid.shape)
    return np.nanmean(stack, axis=0), sd


def main():
    trials = load_trials()
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)

    fig, axes = plt.subplots(1, len(LEVELS), figsize=(3.4 * len(LEVELS), 4.0), sharey=True)
    for ax, slip in zip(axes, LEVELS):
        counts = []
        for subject in SUBJECTS:
            group = [t for t in trials if t['subject'] == subject and t['slip'] == slip]
            if not group:
                continue
            mu, sd = mean_sd(group, grid)
            ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[subject], alpha=0.18, lw=0)
            ax.plot(grid, mu, color=COLORS[subject], lw=2)
            ax.axvline(np.mean([t['pert'] - t['resp'] for t in group]),
                       color=COLORS[subject], lw=1.2, ls='--')
            counts.append(f'{subject} n={len(group)}')
        ax.axvline(0, color='#333333', lw=1.2)
        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.set_xlim(*WINDOW)
        ax.set_title(f'Slip {slip}\n' + ('   '.join(counts) or 'no trials'), loc='left', fontsize=9)
        ax.set_xlabel('Time from ankle response onset (s)')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('Torque sent, right (N·m)')

    handles = [Line2D([], [], color=COLORS[s], lw=2, label=s) for s in SUBJECTS]
    handles += [Line2D([], [], color='#333333', lw=1.2, label='Ankle response onset (ID)'),
                Line2D([], [], color='#777777', lw=1.2, ls='--', label='Perturbation onset (tread1)')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle('Commanded torque, all torque trials (mean ± SD)', y=1.1, fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE, 'figures', 'torque_sent_by_slip.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    summary = pd.DataFrame([dict(subject=t['subject'], trial=t['trial'], slip=t['slip'],
                                 success=int(t['success']), sync=t['sync'],
                                 pert_onset_s=round(t['pert'], 3), response_onset_s=round(t['resp'], 3),
                                 response_delay_s=round(t['resp'] - t['pert'], 3)) for t in trials])
    summary = summary.sort_values(['slip', 'subject', 'trial'])
    print(summary.to_string(index=False))
    out = os.path.join(BASE, 'figures', 'torque_sent_by_slip_onsets.csv')
    summary.to_csv(out, index=False)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
