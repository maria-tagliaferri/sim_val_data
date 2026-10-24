"""Plot AP whole-body COM relative to the base of support (BOS = toes):
COM − BOS displacement, velocity and acceleration, aligned to perturbation
onset, one figure per slip (torque on vs no torque overlaid).

COM: OpenSim whole-body center of mass from the IK coordinates.
BOS: AP position of the toes = mean of the model's RMT1, RMT5, LMT1, LMT5
     markers. The heels are left out because they move on their own
     (heel rise) during the response. These are the model's markers, so this
     still works when the experimental markers are missing.
Positions are low-pass filtered (4th-order zero-lag Butterworth) and then
differentiated for velocity and acceleration. Displacements are the change
from the value at perturbation onset, so relative displacement starts at 0.

Perturbation onsets come from figures/alignment_onsets.csv, written by
plot_aligned_ankle_exo.py (Vicon time = IK time).

Run with the OpenSim env:
    /opt/anaconda3/envs/opensim_scripting/bin/python plot_com_bos.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import opensim as osim
import pandas as pd
from scipy.signal import butter, filtfilt

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, 'H1018_Ank_Exo_scaled_all.osim')
BOS_MARKERS = ['RMT1', 'RMT5', 'LMT1', 'LMT5']  # toes; heels excluded
CUTOFF_HZ = 6.0
WINDOW = (-0.5, 2.5)  # s relative to perturbation onset
PLOT_EXCLUDE_SLIPS = {4, 5}
SUCCESS_ONLY = True
INCLUDE_UNSUCCESSFUL_SLIPS = {11}  # also plot failed trials here (dashed)
COND_COLORS = {'y': '#1f5fd6', 'n': '#2ca02c'}  # torque on = blue, no torque = green
COND_LABELS = {'y': 'Exo torque on', 'n': 'No torque'}
ROWS = [('pos', 100, 'AP displacement (cm)'),
        ('vel', 100, 'AP velocity (cm/s)'),
        ('acc', 1, 'AP acceleration (m/s²)')]


def read_sto(path):
    with open(path) as f:
        n = next(i for i, l in enumerate(f) if l.strip() == 'endheader')
    return pd.read_csv(path, sep=r'\s+', skiprows=n + 1)


def com_bos_ap(model, state, ik):
    """Return AP (ground X) COM and BOS position (m) for every IK frame."""
    coords = model.getCoordinateSet()
    names = [c for c in ik.columns if c != 'time']
    rot = {c: coords.get(c).getMotionType() == osim.Coordinate.Rotational for c in names}
    markers = [model.getMarkerSet().get(m) for m in BOS_MARKERS]
    com, bos = np.empty(len(ik)), np.empty(len(ik))
    for i, row in enumerate(ik[names].values):
        for c, v in zip(names, row):
            coords.get(c).setValue(state, np.deg2rad(v) if rot[c] else v, False)
        model.assemble(state)
        model.realizePosition(state)
        com[i] = model.calcMassCenterPosition(state)[0]
        bos[i] = np.mean([m.getLocationInGround(state)[0] for m in markers])
    return com, bos


def kinematics(t, x):
    fs = 1 / np.median(np.diff(t))
    b, a = butter(4, CUTOFF_HZ / (fs / 2))
    pos = filtfilt(b, a, x)
    vel = np.gradient(pos, t)
    acc = np.gradient(vel, t)
    return dict(pos=pos, vel=vel, acc=acc)


def main():
    info = pd.read_csv(os.path.join(BASE, 'Trial_Info.csv'))
    info.columns = [c.strip() for c in info.columns]
    info = info.set_index('Trial Number')
    onsets = pd.read_csv(os.path.join(BASE, 'figures', 'alignment_onsets.csv'))

    model = osim.Model(MODEL)
    state = model.initSystem()
    grid = np.arange(WINDOW[0], WINDOW[1] + 1e-9, 0.01)

    trials = []
    for _, o in onsets.iterrows():
        n, slip, torque = int(o['trial']), int(o['slip']), o['torque']
        success = int(info.loc[n, 'Success']) == 1
        if slip in PLOT_EXCLUDE_SLIPS or (SUCCESS_ONLY and not success
                                          and slip not in INCLUDE_UNSUCCESSFUL_SLIPS):
            continue
        ik = read_sto(os.path.join(BASE, 'IK', f'inverse_kinematics_Trial_{n}_slip{slip}.mot'))
        com, bos = com_bos_ap(model, state, ik)
        t = ik['time'].values
        tr = dict(n=n, slip=slip, torque=torque, success=success)
        t_rel = t - o['onset_vicon_s']
        for key, x in [('com', com), ('bos', bos)]:
            k = kinematics(t, x)
            k['pos'] = k['pos'] - np.interp(0, t_rel, k['pos'])  # change from onset
            tr[key] = {q: np.interp(grid, t_rel, v, left=np.nan, right=np.nan) for q, v in k.items()}
        trials.append(tr)
        print(f'Trial {n} (slip {slip}, torque {torque}) done')

    # ---------- Plot: one figure per slip, rows = relative pos/vel/acc (COM − BOS) ----------
    slips = sorted({t['slip'] for t in trials})
    figs = []
    for slip in slips:
        group = [t for t in trials if t['slip'] == slip]
        show_fail = any(not t['success'] for t in group)
        fig, axes = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
        for r, (q, scale, ylabel) in enumerate(ROWS):
            ax = axes[r]
            for cond in ['n', 'y']:
                for ok in [True, False]:
                    sub = [t for t in group if t['torque'] == cond and t['success'] == ok]
                    if not sub:
                        continue
                    ls = '-' if ok else '--'
                    label = COND_LABELS[cond] + ((', successful' if ok else ', unsuccessful')
                                                 if show_fail else '')
                    curves = np.array([t['com'][q] - t['bos'][q] for t in sub]) * scale
                    for y in curves:
                        ax.plot(grid, y, color=COND_COLORS[cond], lw=0.8, alpha=0.3, ls=ls)
                    ax.plot(grid, np.nanmean(curves, 0), color=COND_COLORS[cond], lw=2, ls=ls,
                            label=f'{label} (n={len(sub)})')
            ax.axvline(0, color='#888888', lw=1, ls=':')
            ax.axhline(0, color='#cccccc', lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.set_ylabel(f'COM − toes {ylabel}')
        axes[-1].set_xlabel('Time from perturbation onset (s)')
        axes[0].legend(loc='upper right', frameon=False, fontsize=9)
        fig.suptitle(f'Slip {slip}: AP COM relative to base of support (toes)'
                     f'{" (successful trials)" if SUCCESS_ONLY and not show_fail else ""}\n'
                     '(+ = COM anterior of toes; thin = trials, bold = mean)', fontsize=11)
        figs.append((fig, axes, slip))

    # same y-limits across figures so slip levels stay comparable
    for r in range(len(ROWS)):
        lo = min(a[r].get_ylim()[0] for _, a, _ in figs)
        hi = max(a[r].get_ylim()[1] for _, a, _ in figs)
        for _, a, _ in figs:
            a[r].set_ylim(lo, hi)

    for fig, _, slip in figs:
        fig.tight_layout()
        out = os.path.join(BASE, 'figures', f'com_rel_bos_slip_{slip}')
        fig.savefig(out + '.png', dpi=200)
        fig.savefig(out + '.pdf')
        plt.close(fig)
        print(f'Saved {out}.png/.pdf')


if __name__ == '__main__':
    main()
