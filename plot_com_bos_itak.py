"""Itak: AP whole-body COM relative to the base of support (toes), one figure
per slip level (7, 8, 9), torque on vs no torque, successful vs unsuccessful.

Same method as plot_com_bos.py (COM and toe markers from the scaled model,
6 Hz low-pass, displacement = change from onset), applied to Itak/:
    model   Itak/H1018_Ank_ExoItakscaled.osim
    IK      Itak/IK/inverse_kinematics_Trial_*_slip*.mot
    onset   tread1 marker in Itak/Marker/Trial_*_slip*.csv (> 1 mm, Vicon time = IK time)
    outcome Success column of Itak/Itak_trial_info.csv
Trials with no Success entry are drawn dotted as 'outcome not recorded'.

Run with the OpenSim env:
    /opt/anaconda3/envs/opensim_scripting/bin/python plot_com_bos_itak.py
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import opensim as osim
import pandas as pd

from plot_aligned_ankle_exo import read_vicon_csv, tread1_onset
from plot_com_bos import COND_COLORS, COND_LABELS, ROWS, WINDOW, com_bos_ap, kinematics, read_sto

BASE = os.path.dirname(os.path.abspath(__file__))
ITAK = os.path.join(BASE, 'Itak')
MODEL = os.path.join(ITAK, 'H1018_Ank_ExoItakscaled.osim')
SLIPS = (7, 8, 9)
NAME = re.compile(r'inverse_kinematics_Trial_(\d+)_slip(\d+)\.mot')
OUTCOME_STYLE = {True: ('-', 'successful'), False: ('--', 'unsuccessful'),
                 None: (':', 'outcome not recorded')}


def read_outcomes():
    """{trial: True/False/None} from Itak_trial_info.csv (skips the note rows)."""
    info = pd.read_csv(os.path.join(ITAK, 'Itak_trial_info.csv'), dtype=str)
    info.columns = [c.strip() for c in info.columns]
    out = {}
    for _, r in info.iterrows():
        n, ok = str(r['Trial Number']).strip(), str(r['Success']).strip()
        if n.isdigit():
            out[int(n)] = (ok == '1') if ok in ('0', '1') else None
    return out


def main():
    outcomes = read_outcomes()
    model = osim.Model(MODEL)
    state = model.initSystem()
    grid = np.arange(WINDOW[0], WINDOW[1] + 1e-9, 0.01)

    trials = []
    for path in sorted(glob.glob(os.path.join(ITAK, 'IK', 'inverse_kinematics_Trial_*.mot'))):
        n, slip = map(int, NAME.search(os.path.basename(path)).groups())
        if slip not in SLIPS:
            continue
        exo = glob.glob(os.path.join(ITAK, 'IMU', f'Pert_Policy_trial_{n}_slip_{slip}_*_data.csv'))
        if len(exo) != 1:
            print(f'Trial {n} (slip {slip}): {len(exo)} exo logs, cannot tell torque condition, skipped')
            continue
        torque = 'n' if '_notorque_' in os.path.basename(exo[0]) else 'y'
        _, names, traj = read_vicon_csv(os.path.join(ITAK, 'Marker', f'Trial_{n}_slip{slip}.csv'))
        onset = tread1_onset(names, traj)
        if onset is None:
            print(f'Trial {n} (slip {slip}): no tread1 onset, skipped')
            continue
        ik = read_sto(path)
        com, bos = com_bos_ap(model, state, ik)
        t = ik['time'].values
        t_rel = t - onset
        tr = dict(n=n, slip=slip, torque=torque, success=outcomes.get(n))
        for key, x in [('com', com), ('bos', bos)]:
            k = kinematics(t, x)
            k['pos'] = k['pos'] - np.interp(0, t_rel, k['pos'])  # change from onset
            tr[key] = {q: np.interp(grid, t_rel, v, left=np.nan, right=np.nan) for q, v in k.items()}
        trials.append(tr)
        print(f'Trial {n} (slip {slip}, torque {torque}, success {tr["success"]}, '
              f'onset {onset:.2f} s) done')

    figs = []
    for slip in SLIPS:
        group = [t for t in trials if t['slip'] == slip]
        if not group:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
        for r, (q, scale, ylabel) in enumerate(ROWS):
            ax = axes[r]
            for cond in ['n', 'y']:
                for ok in [True, False, None]:
                    sub = [t for t in group if t['torque'] == cond and t['success'] is ok]
                    if not sub:
                        continue
                    ls, ok_label = OUTCOME_STYLE[ok]
                    curves = np.array([t['com'][q] - t['bos'][q] for t in sub]) * scale
                    for y in curves:
                        ax.plot(grid, y, color=COND_COLORS[cond], lw=0.8, alpha=0.3, ls=ls)
                    ax.plot(grid, np.nanmean(curves, 0), color=COND_COLORS[cond], lw=2, ls=ls,
                            label=f'{COND_LABELS[cond]}, {ok_label} (n={len(sub)})')
            ax.axvline(0, color='#888888', lw=1, ls=':')
            ax.axhline(0, color='#cccccc', lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.set_ylabel(f'COM − toes {ylabel}')
        axes[-1].set_xlabel('Time from perturbation onset (s)')
        axes[0].legend(loc='upper right', frameon=False, fontsize=8)
        fig.suptitle(f'Itak, slip {slip}: AP COM relative to base of support (toes)\n'
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
        out = os.path.join(BASE, 'figures', f'itak_com_rel_bos_slip_{slip}')
        fig.savefig(out + '.png', dpi=200)
        fig.savefig(out + '.pdf')
        plt.close(fig)
        print(f'Saved {out}.png/.pdf')


if __name__ == '__main__':
    main()
