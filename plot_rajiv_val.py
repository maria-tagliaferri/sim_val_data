"""Rajiv_Val: joint angles, joint moments and COM relative to the feet per
perturbation level, torque vs no torque, separately for successful and
unsuccessful trials.

Inputs (all in Rajiv_Val/)
    trial sheet  Simulation data collection - Rajiv_Val.csv (level, torque, success)
    IK / ID      IK/inverse_kinematics_Trial_*.mot, ID/inverse_dynamics_Trial_*.sto
    onset        tread1 marker in Markers/Trial_*.csv (> 1 mm, Vicon time = IK/ID time)
    model        H1018_Ank_Exo_scaled.osim (COM and toe markers)

Joints (left/right averaged)
    Ankle  angle + dorsiflexion,            moment + dorsiflexion
    Knee   angle + flexion (sign flipped),  moment + flexion (sign flipped)
    Hip    angle + flexion,                 moment + flexion
    Trunk  angle = pelvis_tilt + lumbar_extension (global pitch, + backward lean),
           moment = lumbar_extension_moment (+ extension)
ID moments are net moments: the model has no exo actuator, so torque-on
ankle moments include the exo torque.

COM relative to feet: same method as plot_com_bos.py (whole-body COM minus
the mean AP position of RMT1/RMT5/LMT1/LMT5, 6 Hz low-pass, displacement
= change from onset).

Torque − no torque differences are computed wherever a level/outcome cell
has both conditions, plus a pooled row per outcome (all levels together,
so level differs between the two conditions).

Threshold comparison: each condition at the level closest to its own
staircase threshold (outcomes.txt: no torque 10.08, torque 11.17), i.e.
no torque at level 10 vs torque at level 11, per outcome. Written as
rajiv_val_threshold_{angles,moments,com_rel_feet}.png and
rajiv_val_com_threshold_torque_minus_no_torque.csv.

Run with the OpenSim env:
    /opt/anaconda3/envs/opensim_scripting/bin/python plot_rajiv_val.py
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import opensim as osim
import pandas as pd
from matplotlib.lines import Line2D

from plot_aligned_ankle_exo import read_vicon_csv, tread1_onset
from plot_com_bos import COND_COLORS, COND_LABELS, ROWS, com_bos_ap, kinematics, read_sto

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'Rajiv_Val')
OUT = os.path.join(BASE, 'figures', 'rajiv_val')
WINDOW = (-0.5, 2.5)  # s relative to perturbation onset
DT = 0.01
JOINT_COLORS = {'y': '#d62728', 'n': '#1f5fd6'}  # same as plot_ik_joint_angles.py
JOINT_LABELS = {'y': 'Torque', 'n': 'No torque'}
OUTCOMES = {True: 'successful', False: 'unsuccessful'}
JOINTS = ('ankle', 'knee', 'hip', 'trunk')
ANGLE_LABELS = {'ankle': 'Ankle angle (deg)\n+ dorsiflexion', 'knee': 'Knee angle (deg)\n+ flexion',
                'hip': 'Hip angle (deg)\n+ flexion', 'trunk': 'Trunk angle (deg)\n+ backward lean'}
MOMENT_LABELS = {'ankle': 'Ankle moment (N·m)\n+ dorsiflexion', 'knee': 'Knee moment (N·m)\n+ flexion',
                 'hip': 'Hip moment (N·m)\n+ flexion', 'trunk': 'Lumbar moment (N·m)\n+ extension'}
SUMMARY_WINDOW = (0.0, 1.5)  # s after onset, for peaks and mean differences
THRESHOLD_LEVEL = {'n': 10.0, 'y': 11.0}  # level closest to each condition's threshold


def angles(ik):
    return dict(ankle=(ik['ankle_angle_r'] + ik['ankle_angle_l']).values / 2,
                knee=-(ik['knee_angle_r'] + ik['knee_angle_l']).values / 2,
                hip=(ik['hip_flexion_r'] + ik['hip_flexion_l']).values / 2,
                trunk=(ik['pelvis_tilt'] + ik['lumbar_extension']).values)


def moments(idd):
    return dict(ankle=(idd['ankle_angle_r_moment'] + idd['ankle_angle_l_moment']).values / 2,
                knee=-(idd['knee_angle_r_moment'] + idd['knee_angle_l_moment']).values / 2,
                hip=(idd['hip_flexion_r_moment'] + idd['hip_flexion_l_moment']).values / 2,
                trunk=idd['lumbar_extension_moment'].values)


def load_trials(grid):
    info = pd.read_csv(os.path.join(DATA, 'Simulation data collection - Rajiv_Val.csv'))
    info.columns = [c.strip() for c in info.columns]
    info = info.set_index('Trial Number')
    model = osim.Model(os.path.join(DATA, 'H1018_Ank_Exo_scaled.osim'))
    state = model.initSystem()

    trials = []
    for path in glob.glob(os.path.join(DATA, 'IK', 'inverse_kinematics_Trial_*.mot')):
        stem = os.path.basename(path)[len('inverse_kinematics_'):-len('.mot')]
        n = int(re.match(r'Trial_(\d+)_', stem).group(1))
        _, names, traj = read_vicon_csv(os.path.join(DATA, 'Markers', stem + '.csv'))
        onset = tread1_onset(names, traj)
        if onset is None:
            print(f'Trial {n}: no tread1 onset, skipped')
            continue
        ik = read_sto(path)
        idd = read_sto(os.path.join(DATA, 'ID', f'inverse_dynamics_{stem}.sto'))
        t_ik, t_id = ik['time'].values - onset, idd['time'].values - onset
        tr = dict(n=n, level=float(info.loc[n, 'Slip Level']),
                  torque=str(info.loc[n, 'Torque']).strip(),
                  success=int(info.loc[n, 'Success']) == 1)
        tr['angle'] = {k: np.interp(grid, t_ik, v, left=np.nan, right=np.nan)
                       for k, v in angles(ik).items()}
        tr['moment'] = {k: np.interp(grid, t_id, v, left=np.nan, right=np.nan)
                        for k, v in moments(idd).items()}
        com, bos = com_bos_ap(model, state, ik)
        rel = {}
        for key, x in [('com', com), ('bos', bos)]:
            k = kinematics(ik['time'].values, x)
            k['pos'] = k['pos'] - np.interp(0, t_ik, k['pos'])  # change from onset
            rel[key] = {q: np.interp(grid, t_ik, v, left=np.nan, right=np.nan) for q, v in k.items()}
        tr['com_rel'] = {q: rel['com'][q] - rel['bos'][q] for q in rel['com']}
        trials.append(tr)
        print(f'Trial {n:2d}  level {tr["level"]:4.1f}  torque {tr["torque"]}  '
              f'success {int(tr["success"])}  onset {onset:.2f} s')
    return sorted(trials, key=lambda t: t['n'])


def draw_group(ax, grid, curves, color, with_sd=True):
    for y in curves:
        ax.plot(grid, y, color=color, lw=0.7, alpha=0.3)
    mu = np.nanmean(curves, 0)
    ax.plot(grid, mu, color=color, lw=2)
    if with_sd and len(curves) > 1:
        sd = np.nanstd(curves, 0, ddof=1)
        ax.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)


def style(ax):
    ax.axvline(0, color='#888888', lw=1, ls=':')
    ax.axhline(0, color='#cccccc', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot_joints(trials, grid, success, kind, labels):
    sub = [t for t in trials if t['success'] == success]
    levels = sorted({t['level'] for t in sub})
    fig, axes = plt.subplots(len(levels), len(JOINTS), figsize=(13, 2.5 * len(levels) + 0.9),
                             sharex=True, sharey='col', squeeze=False)
    for r, level in enumerate(levels):
        counts = []
        for cond in ('n', 'y'):
            group = [t for t in sub if t['level'] == level and t['torque'] == cond]
            counts.append(f'{JOINT_LABELS[cond].lower()} n={len(group)}')
            for c, j in enumerate(JOINTS):
                if group:
                    draw_group(axes[r, c], grid, np.array([t[kind][j] for t in group]),
                               JOINT_COLORS[cond])
        for c, j in enumerate(JOINTS):
            style(axes[r, c])
            axes[r, c].set_ylabel(labels[j], fontsize=8)
            if r == 0:
                axes[r, c].set_title(j.capitalize(), fontsize=11)
        axes[r, 0].set_ylabel(f'Level {level:g}\n({", ".join(counts)})\n\n{labels[JOINTS[0]]}',
                              fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time from perturbation onset (s)')
    handles = [Line2D([], [], color=JOINT_COLORS[c], lw=2, label=JOINT_LABELS[c]) for c in ('y', 'n')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False)
    what = 'Joint angles' if kind == 'angle' else 'Joint moments (net, from ID)'
    fig.suptitle(f'Rajiv_Val, {OUTCOMES[success]} trials: {what}\n'
                 'mean of left and right; bold = mean, band = ±1 SD, thin = trials',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT, f'rajiv_val_{kind}s_{OUTCOMES[success]}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')


def plot_com(trials, grid, success):
    sub = [t for t in trials if t['success'] == success]
    levels = sorted({t['level'] for t in sub})
    fig, axes = plt.subplots(len(ROWS), len(levels), figsize=(3.3 * len(levels) + 0.6, 8),
                             sharex=True, sharey='row', squeeze=False)
    for c, level in enumerate(levels):
        counts = []
        for cond in ('n', 'y'):
            group = [t for t in sub if t['level'] == level and t['torque'] == cond]
            counts.append(f'{"torque" if cond == "y" else "no torque"} n={len(group)}')
            for r, (q, scale, _) in enumerate(ROWS):
                if group:
                    draw_group(axes[r, c], grid, np.array([t['com_rel'][q] for t in group]) * scale,
                               COND_COLORS[cond])
        axes[0, c].set_title(f'Level {level:g}\n{", ".join(counts)}', loc='left', fontsize=9)
        for r in range(len(ROWS)):
            style(axes[r, c])
    for r, (_, _, ylabel) in enumerate(ROWS):
        axes[r, 0].set_ylabel(f'COM − toes\n{ylabel}')
    for ax in axes[-1]:
        ax.set_xlabel('Time from onset (s)')
    handles = [Line2D([], [], color=COND_COLORS[c], lw=2, label=COND_LABELS[c]) for c in ('y', 'n')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False)
    fig.suptitle(f'Rajiv_Val, {OUTCOMES[success]} trials: AP COM relative to the feet (toes)\n'
                 '+ = COM anterior of toes; bold = mean, band = ±1 SD, thin = trials',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT, f'rajiv_val_com_rel_feet_{OUTCOMES[success]}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')


def com_metrics(t, grid):
    w = (grid >= SUMMARY_WINDOW[0]) & (grid <= SUMMARY_WINDOW[1])
    pos, vel = t['com_rel']['pos'][w] * 100, t['com_rel']['vel'][w] * 100
    return dict(peak_disp_cm=np.nanmax(pos), t_peak_disp_s=grid[w][np.nanargmax(pos)],
                peak_vel_cm_s=np.nanmax(vel))


def diff_row(on, off, grid):
    """Torque (on) − no torque (off) COM metrics; empty if either group is missing."""
    if not on or not off:
        return {}
    w = (grid >= SUMMARY_WINDOW[0]) & (grid <= SUMMARY_WINDOW[1])
    m_on = pd.DataFrame([com_metrics(t, grid) for t in on]).mean()
    m_off = pd.DataFrame([com_metrics(t, grid) for t in off]).mean()
    row = {f'diff_{k}': m_on[k] - m_off[k] for k in m_on.index}
    mean_curve = lambda g: np.nanmean([t['com_rel']['pos'] for t in g], 0)[w] * 100
    row['diff_mean_disp_0_1.5s_cm'] = np.nanmean(mean_curve(on) - mean_curve(off))
    return row


def com_differences(trials, grid):
    """Torque − no torque for each level/outcome cell with both conditions, plus pooled rows."""
    per_trial = pd.DataFrame([dict(trial=t['n'], level=t['level'], torque=t['torque'],
                                   success=t['success'], **com_metrics(t, grid)) for t in trials])
    cells = [(lvl, ok) for lvl in sorted(per_trial.level.unique()) for ok in (True, False)]
    cells += [('pooled', True), ('pooled', False)]
    rows = []
    for level, ok in cells:
        sel = [t for t in trials if t['success'] == ok and (level == 'pooled' or t['level'] == level)]
        on = [t for t in sel if t['torque'] == 'y']
        off = [t for t in sel if t['torque'] == 'n']
        row = dict(level=level, outcome=OUTCOMES[ok], n_torque=len(on), n_no_torque=len(off))
        if level == 'pooled':
            row['levels_torque'] = ' '.join(f'{v:g}' for v in sorted({t['level'] for t in on}))
            row['levels_no_torque'] = ' '.join(f'{v:g}' for v in sorted({t['level'] for t in off}))
        row.update(diff_row(on, off, grid))
        rows.append(row)
    return per_trial, pd.DataFrame(rows)


def threshold_groups(trials, success):
    return {c: [t for t in trials if t['success'] == success and t['torque'] == c
                and t['level'] == THRESHOLD_LEVEL[c]] for c in ('n', 'y')}


def threshold_label(cond, n):
    return f'{JOINT_LABELS[cond]}, level {THRESHOLD_LEVEL[cond]:g} (n={n})'


def plot_threshold(trials, grid):
    """No torque at level 10 vs torque at level 11: joints (angles, moments) and COM."""
    for kind, labels in (('angle', ANGLE_LABELS), ('moment', MOMENT_LABELS)):
        fig, axes = plt.subplots(2, len(JOINTS), figsize=(13, 6.2), sharex=True, sharey='col')
        for r, success in enumerate((True, False)):
            groups = threshold_groups(trials, success)
            for cond, group in groups.items():
                for c, j in enumerate(JOINTS):
                    draw_group(axes[r, c], grid, np.array([t[kind][j] for t in group]),
                               JOINT_COLORS[cond])
            for c, j in enumerate(JOINTS):
                style(axes[r, c])
                axes[r, c].set_ylabel(labels[j], fontsize=8)
                if r == 0:
                    axes[r, c].set_title(j.capitalize(), fontsize=11)
            counts = ', '.join(f'{JOINT_LABELS[c].lower()} n={len(g)}' for c, g in groups.items())
            axes[r, 0].set_ylabel(f'{OUTCOMES[success].capitalize()}\n({counts})\n\n'
                                  f'{labels[JOINTS[0]]}', fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel('Time from perturbation onset (s)')
        handles = [Line2D([], [], color=JOINT_COLORS[c], lw=2,
                          label=f'{JOINT_LABELS[c]}, level {THRESHOLD_LEVEL[c]:g}') for c in ('y', 'n')]
        fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False)
        what = 'Joint angles' if kind == 'angle' else 'Joint moments (net, from ID)'
        fig.suptitle(f'Rajiv_Val at each condition\'s own threshold: {what}\n'
                     'mean of left and right; bold = mean, band = ±1 SD, thin = trials',
                     x=0.01, ha='left', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = os.path.join(OUT, f'rajiv_val_threshold_{kind}s.png')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'Saved {out}')

    fig, axes = plt.subplots(len(ROWS), 2, figsize=(10, 8), sharex=True, sharey='row')
    for c, success in enumerate((True, False)):
        groups = threshold_groups(trials, success)
        for cond, group in groups.items():
            for r, (q, scale, _) in enumerate(ROWS):
                draw_group(axes[r, c], grid, np.array([t['com_rel'][q] for t in group]) * scale,
                           COND_COLORS[cond])
        axes[0, c].set_title(f'{OUTCOMES[success].capitalize()}\n'
                             + ', '.join(threshold_label(k, len(g)) for k, g in groups.items()),
                             loc='left', fontsize=9)
        for r in range(len(ROWS)):
            style(axes[r, c])
    for r, (_, _, ylabel) in enumerate(ROWS):
        axes[r, 0].set_ylabel(f'COM − toes\n{ylabel}')
    for ax in axes[-1]:
        ax.set_xlabel('Time from onset (s)')
    handles = [Line2D([], [], color=COND_COLORS[c], lw=2,
                      label=f'{COND_LABELS[c]}, level {THRESHOLD_LEVEL[c]:g}') for c in ('y', 'n')]
    fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False)
    fig.suptitle('Rajiv_Val at each condition\'s own threshold: AP COM relative to the feet\n'
                 '+ = COM anterior of toes; bold = mean, band = ±1 SD, thin = trials',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUT, 'rajiv_val_threshold_com_rel_feet.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')

    rows = []
    for success in (True, False):
        g = threshold_groups(trials, success)
        rows.append(dict(outcome=OUTCOMES[success], n_torque_L11=len(g['y']),
                         n_no_torque_L10=len(g['n']), **diff_row(g['y'], g['n'], grid)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    grid = np.arange(WINDOW[0], WINDOW[1] + DT / 2, DT)
    trials = load_trials(grid)
    for success in (True, False):
        plot_joints(trials, grid, success, 'angle', ANGLE_LABELS)
        plot_joints(trials, grid, success, 'moment', MOMENT_LABELS)
        plot_com(trials, grid, success)

    per_trial, diffs = com_differences(trials, grid)
    per_trial.to_csv(os.path.join(OUT, 'rajiv_val_com_metrics_per_trial.csv'), index=False)
    diffs.to_csv(os.path.join(OUT, 'rajiv_val_com_torque_minus_no_torque.csv'), index=False)
    pd.set_option('display.width', 250)
    print('\nPer-trial COM − toes metrics (0–1.5 s after onset), mean by cell:')
    print(per_trial.groupby(['level', 'success', 'torque'])[['peak_disp_cm', 't_peak_disp_s', 'peak_vel_cm_s']]
          .agg(['count', 'mean', 'std']).to_string(float_format=lambda x: f'{x:6.2f}'))
    print('\nTorque − no torque:')
    print(diffs.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))

    thr = plot_threshold(trials, grid)
    thr.to_csv(os.path.join(OUT, 'rajiv_val_com_threshold_torque_minus_no_torque.csv'), index=False)
    print('\nAt own threshold, torque (level 11) − no torque (level 10):')
    print(thr.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))


if __name__ == '__main__':
    main()
