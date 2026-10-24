"""AP centre-of-mass state relative to the centre of the feet, per subject and
slip level, torque on vs no torque.

COM:  whole-body COM from each subject's scaled model at every IK frame
      (OpenSim calcMassCenterPosition = mass-weighted sum of segment COMs).
Feet: mean AP position of the model's RHEEL, RMT1, RMT5, LHEEL, LMT1, LMT5
      markers (model markers, so this works when experimental markers drop out).
Relative position = COM - feet centre (ground X, + = COM anterior), low-pass
filtered (4th-order zero-lag Butterworth, 6 Hz) and differentiated for
velocity and acceleration. Displacement is the change from perturbation onset.

Perturbation onset = tread1 onset (Vicon time = IK time). Trials with no
tread1 data (Maria 8-13) are skipped.
Slip level comes from the IK filename; torque / success from the trial info.
Successful trials: mean +/- SD per condition. Unsuccessful trials: thin dashed.

Run with the OpenSim env:
    /opt/anaconda3/envs/opensim_scripting/bin/python plot_com_state.py
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import opensim as osim
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt

from plot_aligned_ankle_exo import read_sto, read_vicon_csv, tread1_onset

BASE = os.path.dirname(os.path.abspath(__file__))
# label -> (subject folder, scaled model, marker folder)
SUBJECTS = {
    'Maria': ('Maria_Val', 'H1018_Ank_Exo_scaled_all.osim', 'Marker'),  # model sits in the project root
    # Rafay ('Rafay_Val', 'Rafay_Val/H1018_Ank_Exo_scaled.osim', 'Marker') left out for now
    'Rajiv': ('Rajiv_Val', 'Rajiv_Val/H1018_Ank_Exo_scaled.osim', 'Markers'),
    'Itak': ('Itak_Val', 'Itak_Val/H1018_Ank_ExoItakscaled.osim', 'Marker'),
}
LEVELS = (7, 8, 9, 9.5, 10, 10.5, 11, 11.5)
FEET_MARKERS = ['RHEEL', 'RMT1', 'RMT5', 'LHEEL', 'LMT1', 'LMT5']
CUTOFF_HZ = 6.0
WINDOW = (-0.5, 2.0)  # s relative to perturbation onset
DT = 0.01
COND_COLORS = {'y': '#0072B2', 'n': '#E69F00'}
COND_LABELS = {'y': 'Torque on', 'n': 'No torque'}
ROWS = [('disp', 100, 'COM − feet\ndisplacement (cm)'),
        ('vel', 100, 'COM − feet\nvelocity (cm/s)'),
        ('acc', 1, 'COM − feet\nacceleration (m/s²)')]
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


def com_rel_feet(model, state, ik):
    """AP (ground X) COM minus feet centre, in m, for every IK frame."""
    coords = model.getCoordinateSet()
    names = [c for c in ik.columns if c != 'time']
    rot = [coords.get(c).getMotionType() == osim.Coordinate.Rotational for c in names]
    markers = [model.getMarkerSet().get(m) for m in FEET_MARKERS]
    rel = np.empty(len(ik))
    for i, row in enumerate(ik[names].values):
        for c, r, v in zip(names, rot, row):
            coords.get(c).setValue(state, np.deg2rad(v) if r else v, False)
        model.realizePosition(state)
        feet = np.mean([m.getLocationInGround(state)[0] for m in markers])
        rel[i] = model.calcMassCenterPosition(state)[0] - feet
    return rel


def pert_onsets(root, marker_dir, trial_ids):
    """tread1 onset per trial (Vicon s); None when tread1 is missing."""
    onsets = {}
    for n in trial_ids:
        marker = glob.glob(os.path.join(BASE, root, marker_dir, f'Trial_{n}_slip*.csv'))
        if marker:
            _, names, traj = read_vicon_csv(marker[0])
            onsets[n] = tread1_onset(names, traj)
    return onsets


def load_subject(subject, root, model_path, marker_dir, grid):
    info = load_info(root)
    files = []
    for path in glob.glob(os.path.join(BASE, root, 'IK', 'inverse_kinematics_Trial_*.mot')):
        m = NAME.search(os.path.basename(path))
        slip = float(m.group(2).replace('p', '.'))
        n = int(m.group(1))
        if slip in LEVELS and n in info.index and info.loc[n, 'Torque'] in COND_COLORS:
            files.append((n, slip, path))
    onsets = pert_onsets(root, marker_dir, [n for n, _, _ in files])

    model = osim.Model(os.path.join(BASE, model_path))
    state = model.initSystem()
    trials = []
    for n, slip, path in sorted(files):
        if onsets.get(n) is None:
            print(f'  {subject} trial {n} (slip {slip:g}): no tread1 onset, skipped')
            continue
        ik = read_sto(path)
        t = ik['time'].values - onsets[n]
        b, a = butter(4, CUTOFF_HZ / (0.5 / np.median(np.diff(t))))
        pos = filtfilt(b, a, com_rel_feet(model, state, ik))
        vel = np.gradient(pos, t)
        acc = np.gradient(vel, t)
        pos = pos - np.interp(0, t, pos)
        trials.append(dict(subject=subject, trial=n, slip=slip, torque=info.loc[n, 'Torque'],
                           success=info.loc[n, 'Success'] == 1,
                           **{k: np.interp(grid, t, v, left=np.nan, right=np.nan)
                              for k, v in (('disp', pos), ('vel', vel), ('acc', acc))}))
    print(f'{subject}: {len(trials)} trials')
    return trials


def plot_subject(subject, trials, grid):
    levels = [lv for lv in LEVELS if any(t['slip'] == lv for t in trials)]
    fig, axes = plt.subplots(len(ROWS), len(levels), figsize=(3.0 * len(levels) + 1, 7.5),
                             sharex=True, sharey='row', squeeze=False)
    for c, slip in enumerate(levels):
        counts = []
        for cond in ('n', 'y'):
            ok = [t for t in trials if t['slip'] == slip and t['torque'] == cond and t['success']]
            fail = [t for t in trials if t['slip'] == slip and t['torque'] == cond and not t['success']]
            if ok or fail:
                counts.append(f'{COND_LABELS[cond]} {len(ok)}' + (f' (+{len(fail)} fail)' if fail else ''))
            for r, (key, scale, _) in enumerate(ROWS):
                ax = axes[r, c]
                for t in fail:
                    ax.plot(grid, t[key] * scale, color=COND_COLORS[cond], lw=0.8, ls='--', alpha=0.6)
                if ok:
                    stack = np.vstack([t[key] for t in ok]) * scale
                    mu = np.nanmean(stack, axis=0)
                    sd = np.nanstd(stack, axis=0, ddof=1) if len(ok) > 1 else np.zeros(grid.shape)
                    ax.fill_between(grid, mu - sd, mu + sd, color=COND_COLORS[cond], alpha=0.2, lw=0)
                    ax.plot(grid, mu, color=COND_COLORS[cond], lw=2)
        axes[0, c].set_title(f'Slip {slip:g}\n' + '\n'.join(counts), loc='left', fontsize=8)
        for r in range(len(ROWS)):
            ax = axes[r, c]
            ax.axvline(0, color='#888888', lw=0.8, ls=':')
            ax.axhline(0, color='#cccccc', lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        axes[-1, c].set_xlabel('Time from perturbation (s)')
    for r, (_, _, label) in enumerate(ROWS):
        axes[r, 0].set_ylabel(label)
    axes[0, 0].set_xlim(*WINDOW)
    handles = [Line2D([], [], color=COND_COLORS[c], lw=2, label=f'{COND_LABELS[c]}, successful (mean ± SD)')
               for c in ('y', 'n')]
    handles += [Line2D([], [], color='#777777', lw=0.8, ls='--', label='Unsuccessful trial')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'{subject}: AP COM relative to centre of feet (+ = COM anterior)', y=1.04, fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE, 'figures', f'com_state_{subject.lower()}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def main():
    osim.Logger.setLevelString('error')
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    out_dir = os.path.join(BASE, 'figures', 'com_state')
    os.makedirs(out_dir, exist_ok=True)
    for subject, (root, model_path, marker_dir) in SUBJECTS.items():
        trials = load_subject(subject, root, model_path, marker_dir, grid)
        plot_subject(subject, trials, grid)
        rows = [pd.DataFrame(dict(subject=subject, trial=t['trial'], slip=t['slip'], torque=t['torque'],
                                  success=int(t['success']), time_from_pert_s=np.round(grid, 3),
                                  disp_cm=t['disp'] * 100, vel_cm_s=t['vel'] * 100, acc_m_s2=t['acc']))
                for t in trials]
        out = os.path.join(out_dir, f'com_state_{subject.lower()}.csv')
        pd.concat(rows).to_csv(out, index=False, float_format='%.4f')
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
