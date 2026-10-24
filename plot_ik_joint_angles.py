"""Sagittal joint angles (IK) averaged across left and right, aligned to
perturbation onset, torque on vs no torque for each slip level.

Angles
------
- Ankle = mean(ankle_angle_r, ankle_angle_l)        + = dorsiflexion
- Knee  = -mean(knee_angle_r, knee_angle_l)         + = flexion (model is - = flexion)
- Hip   = mean(hip_flexion_r, hip_flexion_l)        + = flexion
- Trunk = pelvis_tilt + lumbar_extension            + = backward lean (global trunk pitch)

Perturbation onset
------------------
Same as plot_aligned_ankle_exo.py: first frame where tread1 X moves > 1 mm,
falling back to sim_test/Trial_*.txt + the mean (tread1 - sim_test) offset.
IK time 0 = Vicon frame 1, so the onset is subtracted directly.

Only successful trials (Success column of Trial_Info.csv) are plotted and
averaged; unsuccessful trials include stepping responses.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_aligned_ankle_exo import read_sto, read_vicon_csv, tread1_onset

BASE = os.path.dirname(os.path.abspath(__file__))
WINDOW = (-0.5, 2.5)  # s relative to perturbation onset
DT = 0.01
COLORS = {'y': '#d62728', 'n': '#1f5fd6'}  # torque = red, no torque = blue
LABELS = {'y': 'Torque', 'n': 'No torque'}
JOINTS = (('ankle', 'Ankle (deg)\n+ dorsiflexion'),
          ('knee', 'Knee (deg)\n+ flexion'),
          ('hip', 'Hip (deg)\n+ flexion'),
          ('trunk', 'Trunk (deg)\n+ backward lean'))
NAME = re.compile(r'inverse_kinematics_Trial_(\d+)_slip(\d+)\.mot')


def joint_angles(ik):
    return dict(
        ankle=(ik['ankle_angle_r'] + ik['ankle_angle_l']).values / 2,
        knee=-(ik['knee_angle_r'] + ik['knee_angle_l']).values / 2,
        hip=(ik['hip_flexion_r'] + ik['hip_flexion_l']).values / 2,
        trunk=(ik['pelvis_tilt'] + ik['lumbar_extension']).values)


def load_trials():
    info = pd.read_csv(os.path.join(BASE, 'Trial_Info.csv'))
    info.columns = [c.strip() for c in info.columns]
    info = info.set_index('Trial Number')

    trials = []
    for path in glob.glob(os.path.join(BASE, 'IK', 'inverse_kinematics_Trial_*.mot')):
        n, slip = map(int, NAME.search(os.path.basename(path)).groups())
        name = f'Trial_{n}_slip{slip}'
        marker = os.path.join(BASE, 'Marker', name + '.csv')
        sim = os.path.join(BASE, 'sim_test', name + '.txt')
        tread = None
        if os.path.exists(marker):
            _, names, traj = read_vicon_csv(marker)
            tread = tread1_onset(names, traj)
        ik = read_sto(path)
        trials.append(dict(
            n=n, slip=slip, torque=str(info.loc[n, 'Torque']).strip(),
            success=int(info.loc[n, 'Success']) == 1, tread=tread,
            sim=float(open(sim).read()) if os.path.exists(sim) else None,
            t=ik['time'].values, **joint_angles(ik)))

    diffs = [t['tread'] - t['sim'] for t in trials if t['tread'] is not None and t['sim'] is not None]
    offset = float(np.mean(diffs))
    kept = []
    for t in sorted(trials, key=lambda t: t['n']):
        if t['tread'] is not None:
            onset, src = t['tread'], 'tread1'
        elif t['sim'] is not None:
            onset, src = t['sim'] + offset, 'sim_test+offset'
        else:
            print(f'  Trial {t["n"]} (slip {t["slip"]}): no onset, skipped')
            continue
        t['t'] = t['t'] - onset
        print(f'  Trial {t["n"]:2d}  slip {t["slip"]:2d}  torque {t["torque"]}  '
              f'success {int(t["success"])}  onset {onset:.3f} s ({src})')
        kept.append(t)
    return kept


def plot(trials, title, out):
    # Only slips that have both conditions can be compared.
    slips = sorted(s for s in {t['slip'] for t in trials}
                   if {t['torque'] for t in trials if t['slip'] == s} >= {'y', 'n'})
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    fig, axes = plt.subplots(len(slips), len(JOINTS), figsize=(13, 2.6 * len(slips) + 0.8),
                             sharex=True, sharey='col', squeeze=False)
    for r, slip in enumerate(slips):
        for c, (key, ylabel) in enumerate(JOINTS):
            ax = axes[r, c]
            for cond in ('n', 'y'):
                group = [t for t in trials if t['slip'] == slip and t['torque'] == cond]
                if not group:
                    continue
                curves = np.array([np.interp(grid, t['t'], t[key], left=np.nan, right=np.nan)
                                   for t in group])
                for t, y in zip(group, curves):
                    ax.plot(grid, y, color=COLORS[cond], lw=0.7, alpha=0.3)
                mu = np.nanmean(curves, 0)
                ax.plot(grid, mu, color=COLORS[cond], lw=2)
                if len(group) > 1:
                    sd = np.nanstd(curves, 0, ddof=1)
                    ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[cond], alpha=0.12, lw=0)
            ax.axvline(0, color='#888888', lw=1, ls=':')
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(ylabel.split('\n')[0].replace(' (deg)', ''), fontsize=11)
            if c == 0:
                n_on = sum(t['slip'] == slip and t['torque'] == 'y' for t in trials)
                n_off = sum(t['slip'] == slip and t['torque'] == 'n' for t in trials)
                ax.set_ylabel(f'Slip {slip}\n(torque n={n_on}, no torque n={n_off})\n\n'
                              f'{ylabel}', fontsize=9)
            else:
                ax.set_ylabel(ylabel, fontsize=9)
            if r == len(slips) - 1:
                ax.set_xlabel('Time from perturbation onset (s)')
    handles = [Line2D([], [], color=COLORS['y'], lw=2, label=LABELS['y']),
               Line2D([], [], color=COLORS['n'], lw=2, label=LABELS['n'])]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False)
    fig.suptitle(f'{title}\nmean of left and right (trunk: global pitch); '
                 'bold = mean, band = ±1 SD, thin = trials', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')


def main():
    trials = [t for t in load_trials() if t['success']]
    os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)
    plot(trials, 'Joint angles, successful trials',
         os.path.join(BASE, 'figures', 'ik_joint_angles_torque_vs_none.png'))


if __name__ == '__main__':
    main()
