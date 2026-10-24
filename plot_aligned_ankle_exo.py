"""Time-align exoskeleton (IMU_Torque) data with motion capture (Vicon) data and
plot right ankle angle (IK) / ID ankle moment and exo torque per
perturbation magnitude.

Alignment
---------
- Vicon: rising edge of the analog trigger channel (last column of the Devices
  section of Marker/Trial_*.csv, 1000 Hz).  Vicon time 0 = frame 1, which is
  also time 0 of the IK/ID results.
- Exo:   rising edge of gpio_output in IMU_Torque/Pert_Policy_trial_*.csv.
  t_vicon = t_exo - t_gpio_rise + t_vicon_trigger_rise

Perturbation onset
------------------
- First frame where the tread1 marker X deviates > 1 mm from its initial
  (median of first 50 frames) position.
- If tread1 is missing, use sim_test/Trial_*.txt corrected by the mean
  (tread1 onset - sim_test onset) over the trials that have both.

Biological moment
-----------------
The ID model has no exo actuator, so ID ankle_angle_r_moment is the net moment
(biological + exo).  torque_measured_R uses the OpenSim sign convention
(negative = plantarflexion, it tracks tau_policy_nm), so
    biological = ID net - torque_measured_R (interpolated onto the ID time base).
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRIG_THRESHOLD_V = 1.5
TREAD_THRESHOLD_MM = 1.0
WINDOW = (-0.5, 2.5)  # s relative to perturbation onset
COLORS = {'y': '#d62728', 'n': '#1f5fd6'}  # torque = red, no torque = blue
STYLES = {'y': '-', 'n': '--'}
PLOT_EXCLUDE_SLIPS = {5}
NO_TORQUE_COLOR = '#2ca02c'  # green
ROW_COLORS = [COLORS['n'], COLORS['y']]  # ankle angle = blue, exo torque = red
LABELS = {'y': 'Exo torque on', 'n': 'No torque'}


def read_vicon_csv(path):
    """Return (analog device array, trajectory marker names row, trajectory array)."""
    with open(path, encoding='utf-8-sig', errors='ignore') as f:
        lines = f.read().splitlines()
    ti = next(i for i, l in enumerate(lines) if l.strip() == 'Trajectories')

    def to_array(rows):
        return np.array([[float(x) if x else np.nan for x in r.split(',')]
                         for r in rows if r.strip()])
    devices = to_array(lines[5:ti])
    names = lines[ti + 2].split(',')
    traj = to_array(lines[ti + 5:])
    return devices, names, traj


def vicon_trigger_time(devices):
    t = (devices[:, 0] - 1) / 100 + devices[:, 1] / 1000
    high = devices[:, -1] > TRIG_THRESHOLD_V
    return t[np.argmax(high)] if high.any() else None


def tread1_onset(names, traj):
    cols = [i for i, s in enumerate(names) if s.endswith(':tread1')]
    if not cols:
        return None
    x = traj[:, cols[0]]
    t = (traj[:, 0] - 1) / 100
    ok = ~np.isnan(x)
    if ok.sum() < 50:
        return None
    x0 = np.median(x[ok][:50])
    moved = np.where(ok & (np.abs(x - x0) > TREAD_THRESHOLD_MM))[0]
    if len(moved) == 0 or not ok[:moved[0]].all():
        return None  # never moves, or gap before movement -> onset unreliable
    return t[moved[0]]


def read_sto(path):
    with open(path) as f:
        n = next(i for i, l in enumerate(f) if l.strip() == 'endheader')
    return pd.read_csv(path, sep=r'\s+', skiprows=n + 1)


def plot_per_slip(kept, top, top_label, include_no_torque, prefix, unsuccessful_no_torque_slips=()):
    """One figure per slip, successful trials only: top row = `top` (time key, value key),
    bottom row = exo torque. No-torque trials (green) are added to the top row if requested;
    for slips in `unsuccessful_no_torque_slips`, unsuccessful no-torque trials are added too
    (dashed)."""
    slips = sorted({t['slip'] for t in kept} - PLOT_EXCLUDE_SLIPS)
    grid = np.arange(WINDOW[0], WINDOW[1] + 1e-9, 0.01)
    rows_spec = [top, ('exo_t', 'exo_tau')]
    figs = []
    for slip in slips:
        group = [t for t in kept if t['slip'] == slip and t['success']]  # successful trials only
        on = [t for t in group if t['torque'] == 'y']
        off = [t for t in group if t['torque'] == 'n']
        off_fail = ([t for t in kept if t['slip'] == slip and t['torque'] == 'n' and not t['success']]
                    if include_no_torque and slip in unsuccessful_no_torque_slips else [])
        fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
        for r, (tkey, ykey) in enumerate(rows_spec):
            ax = axes[r]
            # top row: optional no-torque (green) + torque-on (blue); exo torque: torque-on only
            groups = [(on, ROW_COLORS[r], LABELS['y'], '-')]
            if r == 0 and include_no_torque:
                groups[:0] = [(off, NO_TORQUE_COLOR, LABELS['n'], '-'),
                              (off_fail, NO_TORQUE_COLOR, LABELS['n'] + ', unsuccessful', '--')]
            for sub, color, label, ls in groups:
                if not sub:
                    continue
                curves = np.array([np.interp(grid, t[tkey], t[ykey], left=np.nan, right=np.nan)
                                   for t in sub])
                for y in curves:
                    ax.plot(grid, y, color=color, lw=0.8, alpha=0.3, ls=ls)
                ax.plot(grid, np.nanmean(curves, 0), color=color, lw=2, ls=ls,
                        label=f'{label} (n={len(sub)})')
            ax.axvline(0, color='#888888', lw=1, ls=':')
            ax.axhline(0, color='#cccccc', lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel(top_label)
        axes[1].set_ylabel('Exo torque, R (N·m)\n(measured)')
        axes[1].set_xlabel('Time from perturbation onset (s)')
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(loc='lower right', frameon=False, fontsize=9)
        fig.suptitle(f'Slip {slip}: right ankle, aligned to perturbation onset'
                     f'{"" if off_fail else " (successful trials)"}\n'
                     '(− = plantarflexion; thin = trials, bold = mean)', fontsize=11)
        figs.append((slip, fig, axes))

    # same y-limits across figures so slip levels stay comparable
    for r in range(len(rows_spec)):
        lo = min(a[r].get_ylim()[0] for _, _, a in figs)
        hi = max(a[r].get_ylim()[1] for _, _, a in figs)
        for _, _, a in figs:
            a[r].set_ylim(lo, hi)

    for slip, fig, _ in figs:
        fig.tight_layout()
        out = os.path.join(BASE, 'figures', f'{prefix}_slip_{slip}')
        fig.savefig(out + '.png', dpi=200)
        plt.close(fig)
        print(f'Saved {out}.png')


def main():
    info = pd.read_csv(os.path.join(BASE, 'Trial_Info.csv'))
    info.columns = [c.strip() for c in info.columns]
    info = info.set_index('Trial Number')

    trials, ignored = [], []
    for n in info.index:
        slip = int(info.loc[n, 'Slip Level'])
        torque = str(info.loc[n, 'Torque']).strip()
        name = f'Trial_{n}_slip{slip}'
        marker = os.path.join(BASE, 'Marker', name + '.csv')
        id_file = os.path.join(BASE, 'ID', f'inverse_dynamics_{name}.sto')
        ik_file = os.path.join(BASE, 'IK', f'inverse_kinematics_{name}.mot')
        exo = glob.glob(os.path.join(BASE, 'IMU_Torque', f'Pert_Policy_trial_{n}_slip_{slip}_*_data.csv'))
        sim = os.path.join(BASE, 'sim_test', name + '.txt')

        missing = [lbl for lbl, ok in [('marker csv', os.path.exists(marker)),
                                       ('ID .sto', os.path.exists(id_file)),
                                       ('IK .mot', os.path.exists(ik_file)),
                                       ('exo csv', len(exo) == 1)] if not ok]
        if missing:
            ignored.append((n, slip, 'missing ' + ', '.join(missing)))
            continue

        devices, names, traj = read_vicon_csv(marker)
        t_vtrig = vicon_trigger_time(devices)
        d = pd.read_csv(exo[0])
        gpio_high = d['gpio_output'].values > 0.5
        if t_vtrig is None or not gpio_high.any():
            ignored.append((n, slip, 'no trigger edge in Vicon or exo gpio'))
            continue
        t_gpio = d['timestamp'].values[np.argmax(gpio_high)]

        trials.append(dict(
            n=n, slip=slip, torque=torque, success=int(info.loc[n, 'Success']) == 1,
            t_vtrig=t_vtrig, t_gpio=t_gpio,
            tread_onset=tread1_onset(names, traj),
            sim_onset=float(open(sim).read()) if os.path.exists(sim) else None,
            id=read_sto(id_file), ik=read_sto(ik_file), exo=d))

    # Offset between sim_test onset and tread1 onset (Vicon time)
    diffs = [t['tread_onset'] - t['sim_onset'] for t in trials
             if t['tread_onset'] is not None and t['sim_onset'] is not None]
    offset = float(np.mean(diffs))
    print(f'tread1 - sim_test onset offset: mean {offset:.4f} s, SD {np.std(diffs):.4f} s, n={len(diffs)}')

    kept = []
    for t in trials:
        if t['tread_onset'] is not None:
            t['onset'], t['onset_src'] = t['tread_onset'], 'tread1'
        elif t['sim_onset'] is not None:
            t['onset'], t['onset_src'] = t['sim_onset'] + offset, 'sim_test+offset'
        else:
            ignored.append((t['n'], t['slip'], 'no tread1 and no sim_test onset'))
            continue
        kept.append(t)

    rows = []
    for t in kept:
        # Exo time -> Vicon time -> relative to perturbation onset
        exo_t = t['exo']['timestamp'].values - t['t_gpio'] + t['t_vtrig'] - t['onset']
        exo_tau = t['exo']['torque_measured_R'].values
        id_t = t['id']['time'].values - t['onset']
        net = t['id']['ankle_angle_r_moment'].values
        exo_on_id = np.interp(id_t, exo_t, exo_tau, left=np.nan, right=np.nan)
        t.update(exo_t=exo_t, exo_tau=exo_tau, id_t=id_t, net=net, bio=net - exo_on_id,
                 ik_t=t['ik']['time'].values - t['onset'], ankle=t['ik']['ankle_angle_r'].values)
        rows.append(dict(trial=t['n'], slip=t['slip'], torque=t['torque'],
                         vicon_trigger_s=t['t_vtrig'], exo_gpio_s=round(t['t_gpio'], 4),
                         onset_vicon_s=round(t['onset'], 4), onset_source=t['onset_src']))
    summary = pd.DataFrame(rows).sort_values('trial')
    print(summary.to_string(index=False))

    ignored.sort()
    print('\nIgnored trials:')
    for n, slip, why in ignored:
        print(f'  Trial {n} (slip {slip}): {why}')

    # ---------- Plots: one figure per slip, rows = (ankle angle | ID moment) / exo torque ----------
    plot_per_slip(kept, ('ik_t', 'ankle'), 'R ankle angle (deg)\n(IK, + = dorsiflexion)',
                  include_no_torque=True, prefix='aligned_ankle_angle_exo_torque')
    plot_per_slip(kept, ('id_t', 'net'), 'R ankle moment, ID (N·m)\n(net = biological + exo)',
                  include_no_torque=True, prefix='aligned_id_ankle_exo_torque',
                  unsuccessful_no_torque_slips={11})
    summary.to_csv(os.path.join(BASE, 'figures', 'alignment_onsets.csv'), index=False)


if __name__ == '__main__':
    main()
