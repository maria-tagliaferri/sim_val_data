"""Joint kinematics (IK) and kinetics (ID) for successful trials, mean +/- SD
per subject and condition, one figure per slip level.

Rows: ankle, knee, hip. Columns: right angle, right moment (left and right
are essentially symmetric in successful trials). Subjects are colour-coded; torque on = solid, no torque = dashed.
Moments are divided by the scaled model's mass (N·m/kg). The ID model has no
exo actuator, so on torque-on trials the ankle moment is biological + exo.

OpenSim sign convention: ankle + = dorsiflexion, knee − = flexion,
hip + = flexion; moments follow the same signs (ankle − = plantarflexor).

Time 0 = tread1 onset (Vicon time = IK/ID time); trials with no tread1 data
are skipped. Slip level comes from the IK filename, torque / success from the
trial info. Angles and moments are low-pass filtered (4th-order zero-lag
Butterworth, CUTOFF_HZ) before averaging.
"""
import glob
import os
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt

from plot_aligned_ankle_exo import read_sto, read_vicon_csv, tread1_onset

BASE = os.path.dirname(os.path.abspath(__file__))
# label -> (subject folder, scaled model, marker folder)
SUBJECTS = {
    'Maria': ('Maria_Val', 'H1018_Ank_Exo_scaled_all.osim', 'Marker'),
    # Rafay ('Rafay_Val', 'Rafay_Val/H1018_Ank_Exo_scaled.osim', 'Marker') left out for now
    'Rajiv': ('Rajiv_Val', 'Rajiv_Val/H1018_Ank_Exo_scaled.osim', 'Markers'),
    'Itak': ('Itak_Val', 'Itak_Val/H1018_Ank_ExoItakscaled.osim', 'Marker'),
}
COLORS = {'Maria': '#0072B2', 'Rafay': '#D55E00', 'Rajiv': '#009E73', 'Itak': '#CC79A7'}
COND_STYLE = {'y': '-', 'n': '--'}
COND_LABELS = {'y': 'Torque on', 'n': 'No torque'}
LEVELS = (7, 8, 9, 9.5, 10, 10.5, 11, 11.5)
JOINTS = [('ankle_angle', 'Ankle'), ('knee_angle', 'Knee'), ('hip_flexion', 'Hip')]
COLUMNS = [('angle', 'r', 'Right angle (deg)'), ('moment', 'r', 'Right moment (N·m/kg)')]
CUTOFF_HZ = 10.0
WINDOW = (-0.3, 1.5)  # s relative to perturbation onset
DT = 0.01
NAME = re.compile(r'inverse_kinematics_Trial_(\d+)_slip(?:slip)?(\d+(?:p\d+)?)\.mot$')


def load_info(root):
    path = [p for p in glob.glob(os.path.join(BASE, root, '*.csv')) if p.lower().endswith('trial_info.csv')][0]
    info = pd.read_csv(path)
    info.columns = [c.strip() for c in info.columns]
    info['Trial Number'] = pd.to_numeric(info['Trial Number'], errors='coerce')
    info = info.dropna(subset=['Trial Number']).set_index('Trial Number')
    info['Torque'] = info['Torque'].astype(str).str.strip()
    info['Success'] = pd.to_numeric(info['Success'], errors='coerce')
    return info


def model_mass(path):
    return sum(float(b.find('mass').text) for b in ET.parse(path).getroot().iter('Body'))


def filtered(t, x):
    b, a = butter(4, CUTOFF_HZ / (0.5 / np.median(np.diff(t))))
    return filtfilt(b, a, x)


def load_trials(grid):
    trials = []
    for subject, (root, model_path, marker_dir) in SUBJECTS.items():
        info = load_info(root)
        mass = model_mass(os.path.join(BASE, model_path))
        for path in sorted(glob.glob(os.path.join(BASE, root, 'IK', 'inverse_kinematics_Trial_*.mot'))):
            m = NAME.search(os.path.basename(path))
            n, slip = int(m.group(1)), float(m.group(2).replace('p', '.'))
            if slip not in LEVELS or n not in info.index or info.loc[n, 'Success'] != 1 \
                    or info.loc[n, 'Torque'] not in COND_STYLE:
                continue
            id_file = glob.glob(os.path.join(BASE, root, 'ID', f'inverse_dynamics_Trial_{n}_slip*.sto'))
            marker = glob.glob(os.path.join(BASE, root, marker_dir, f'Trial_{n}_slip*.csv'))
            if not (id_file and marker):
                print(f'  {subject} trial {n} (slip {slip:g}): missing ID or marker file, skipped')
                continue
            _, names, traj = read_vicon_csv(marker[0])
            onset = tread1_onset(names, traj)
            if onset is None:
                print(f'  {subject} trial {n} (slip {slip:g}): no tread1 onset, skipped')
                continue
            ik, idd = read_sto(path), read_sto(id_file[0])
            t_ik, t_id = ik['time'].values - onset, idd['time'].values - onset
            tr = dict(subject=subject, trial=n, slip=slip, torque=info.loc[n, 'Torque'])
            for joint, _ in JOINTS:
                for side in {side for _, side, _ in COLUMNS}:
                    tr[('angle', joint, side)] = np.interp(
                        grid, t_ik, filtered(t_ik, ik[f'{joint}_{side}'].values), left=np.nan, right=np.nan)
                    tr[('moment', joint, side)] = np.interp(
                        grid, t_id, filtered(t_id, idd[f'{joint}_{side}_moment'].values) / mass,
                        left=np.nan, right=np.nan)
            trials.append(tr)
        print(f'{subject}: {sum(t["subject"] == subject for t in trials)} successful trials (mass {mass:.0f} kg)')
    return trials


def plot_level(slip, group, grid):
    fig, axes = plt.subplots(len(JOINTS), len(COLUMNS), figsize=(8, 8.5), sharex=True)
    counts = []
    for subject in SUBJECTS:
        for cond in ('y', 'n'):
            sub = [t for t in group if t['subject'] == subject and t['torque'] == cond]
            if not sub:
                continue
            counts.append(f'{subject} {COND_LABELS[cond].lower()} n={len(sub)}')
            for r, (joint, _) in enumerate(JOINTS):
                for c, (kind, side, _) in enumerate(COLUMNS):
                    stack = np.vstack([t[(kind, joint, side)] for t in sub])
                    mu = np.nanmean(stack, axis=0)
                    sd = np.nanstd(stack, axis=0, ddof=1) if len(sub) > 1 else np.zeros(grid.shape)
                    ax = axes[r, c]
                    ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[subject], alpha=0.15, lw=0)
                    ax.plot(grid, mu, color=COLORS[subject], lw=1.8, ls=COND_STYLE[cond])
    for r, (_, joint_label) in enumerate(JOINTS):
        for c, (kind, _, col_label) in enumerate(COLUMNS):
            ax = axes[r, c]
            ax.axvline(0, color='#888888', lw=0.8, ls=':')
            if kind == 'moment':
                ax.axhline(0, color='#cccccc', lw=0.8)
            ax.set_xlim(*WINDOW)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(col_label, fontsize=10)
        axes[r, 0].set_ylabel(joint_label, fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel('Time from perturbation (s)')

    handles = [Line2D([], [], color=COLORS[s], lw=2, label=s) for s in SUBJECTS
               if any(t['subject'] == s for t in group)]
    handles += [Line2D([], [], color='#555555', lw=1.8, ls=COND_STYLE[c], label=COND_LABELS[c])
                for c in ('y', 'n') if any(t['torque'] == c for t in group)]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'Slip {slip:g}: right joint angles and moments, successful trials (mean ± SD)\n'
                 + '   '.join(counts) + '\n+ ankle = dorsiflexion, − knee = flexion, + hip = flexion',
                 y=1.08, fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, 'figures', 'joint_kin_kinetics', f'joint_kin_kinetics_slip_{slip:g}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def main():
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    os.makedirs(os.path.join(BASE, 'figures', 'joint_kin_kinetics'), exist_ok=True)
    trials = load_trials(grid)
    for slip in LEVELS:
        group = [t for t in trials if t['slip'] == slip]
        if group:
            plot_level(slip, group, grid)


if __name__ == '__main__':
    main()
