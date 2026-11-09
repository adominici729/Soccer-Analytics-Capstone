"""
Input:  shots.csv from our supplementary build (soccer_csv/shots.csv)
Output: fig1_distance.png, fig2_shotmap.png, fig3_context.png, fig4_players.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# ---- File paths ----

shots_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('../soccer_csv/shots.csv')
out_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('figs')
out_folder.mkdir(parents=True, exist_ok=True)

# ---- Chart style ----

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'figure.facecolor': 'white',
})
BLUE = '#2a78d6'
ORANGE = '#eb6834'
GRAY = '#52514e'

# ---- Load data and build the analysis set ----

shots = pd.read_csv(shots_path)

# Analysis set: no penalties, no shots straight from corners,
# and no shots where the goalkeeper is missing from the freeze frame (32 shots)
analysis = shots[
    (shots['shot_type'] != 'Penalty')
    & (shots['shot_type'] != 'Corner')
    & shots['gk_x'].notna()
].copy()

overall_rate = analysis['is_goal'].mean()
print('Analysis shots:', len(analysis), '| Goal rate:', round(overall_rate, 4))


# =====================================================================
# Figure B1: shot volume and goal rate by distance
# =====================================================================

bins = np.arange(0, 42, 2)   # 2-unit bins out to 40 units
analysis['distance_bin'] = pd.cut(analysis['distance'], bins)

by_distance = (analysis.groupby('distance_bin', observed=True)['is_goal']
               .agg(['size', 'mean']).reset_index())
bin_middles = [b.mid for b in by_distance['distance_bin']]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

# Left: how many shots in each distance bin
ax[0].bar(bin_middles, by_distance['size'], width=1.8, color=BLUE)
ax[0].set_xlabel('Shot distance to goal centre (coordinate units, 2-unit bins)')
ax[0].set_ylabel('Non-penalty shots')
ax[0].set_title('How many shots', loc='left', fontsize=11)

# Right: share of those shots that were goals
ax[1].plot(bin_middles, by_distance['mean'], color=BLUE, lw=2, marker='o', ms=5)
ax[1].set_xlabel('Shot distance to goal centre (coordinate units, 2-unit bins)')
ax[1].set_ylabel('Share of shots scored')
ax[1].set_title('How often they go in', loc='left', fontsize=11)
ax[1].yaxis.set_major_formatter(PercentFormatter(1.0))
ax[1].axhline(overall_rate, color=GRAY, ls='--', lw=1)
ax[1].text(39, overall_rate + 0.015, f'overall {overall_rate:.1%}', ha='right', color=GRAY, fontsize=9)

fig.tight_layout()
fig.savefig(out_folder / 'fig1_distance.png', dpi=200)
plt.close()


# =====================================================================
# Figure B2: goal rate by shot location (heat map of the attacking half)
# =====================================================================

x_edges = np.arange(60, 121, 4)   # 4 x 4 unit cells
y_edges = np.arange(0, 81, 4)
shot_x = analysis['x'].clip(60, 120)

shot_counts, _, _ = np.histogram2d(shot_x, analysis['y'], bins=[x_edges, y_edges])
goal_counts, _, _ = np.histogram2d(shot_x, analysis['y'], bins=[x_edges, y_edges],
                                   weights=analysis['is_goal'])

# Only show cells with at least 30 shots
goal_rate = np.where(shot_counts >= 30, goal_counts / np.maximum(shot_counts, 1), np.nan)

blues = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
blues.set_bad('#f3f2ee')   # gray for cells with too few shots

fig, ax = plt.subplots(figsize=(7, 4.6))
heat = ax.pcolormesh(y_edges, x_edges, goal_rate, cmap=blues, vmin=0, vmax=0.4,
                     edgecolors='white', linewidth=1)

# Draw the penalty area (x = 102) and goal area (x = 114)
for line_x, left_y, right_y in [(102, 18, 62), (114, 30, 50)]:
    ax.plot([left_y, left_y, right_y, right_y], [120, line_x, line_x, 120], color=GRAY, lw=1)

# Draw the goal between the posts at y = 36 and y = 44
ax.plot([36, 44], [120, 120], color='#0b0b0b', lw=4)

ax.set_xlim(0, 80)
ax.set_ylim(60, 121)
ax.set_aspect('equal')
ax.grid(False)
ax.set_xlabel('Pitch width (y)')
ax.set_ylabel('Pitch length (x), goal at top')

colorbar = fig.colorbar(heat, ax=ax, shrink=0.8)
colorbar.set_label('Share of shots scored')
colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

fig.tight_layout()
fig.savefig(out_folder / 'fig2_shotmap.png', dpi=200)
plt.close()


# =====================================================================
# Figure B3: defenders in the shooting lane, and type of key pass
# =====================================================================

# n_defenders_in_cone is the same as the repo's defenders_in_lane
by_defenders = (analysis.groupby(analysis['n_defenders_in_cone'].clip(upper=4))['is_goal']
                .agg(['size', 'mean']))

# Shots with no key pass get their own category; only keep types with 200+ shots
by_pass = (analysis.assign(assist_type=analysis['assist_type'].fillna('No key pass'))
           .groupby('assist_type')['is_goal'].agg(['size', 'mean']))
by_pass = by_pass[by_pass['size'] >= 200].sort_values('mean')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1, 1.3]})

# Left: defenders in the lane
labels = ['0', '1', '2', '3', '4+']
ax[0].bar(labels, by_defenders['mean'], color=BLUE, width=0.6)
for i, (n, rate) in enumerate(zip(by_defenders['size'], by_defenders['mean'])):
    ax[0].text(i, rate + 0.004, f'{rate:.1%}\nn={n:,}', ha='center', fontsize=8, color=GRAY)
ax[0].set_xlabel('defenders_in_lane (opponents between shooter and posts)')
ax[0].set_ylabel('Share of shots scored')
ax[0].yaxis.set_major_formatter(PercentFormatter(1.0))
ax[0].set_ylim(0, 0.17)
ax[0].set_title('Defenders in the shooting lane', loc='left', fontsize=11)

# Right: type of key pass
ax[1].barh(by_pass.index, by_pass['mean'], color=BLUE, height=0.6)
for i, (n, rate) in enumerate(zip(by_pass['size'], by_pass['mean'])):
    ax[1].text(rate + 0.004, i, f'{rate:.1%} (n={n:,})', va='center', fontsize=8, color=GRAY)
ax[1].set_xlabel('Share of shots scored')
ax[1].xaxis.set_major_formatter(PercentFormatter(1.0))
ax[1].set_xlim(0, 0.38)
ax[1].set_title('Type of key pass before the shot', loc='left', fontsize=11)
ax[1].grid(axis='y', visible=False)

fig.tight_layout()
fig.savefig(out_folder / 'fig3_context.png', dpi=200)
plt.close()


# =====================================================================
# Figure B4: player finishing (goals minus StatsBomb xG vs. shots taken)
# =====================================================================

players = (analysis.groupby(['player_id', 'player'])
           .agg(shots=('is_goal', 'size'), goals=('is_goal', 'sum'), xg=('statsbomb_xg', 'sum'))
           .reset_index())
players['goals_minus_xg'] = players['goals'] - players['xg']

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(players['shots'], players['goals_minus_xg'], s=12, color=BLUE, alpha=0.45, edgecolors='none')
ax.axhline(0, color=GRAY, lw=1)
ax.axvline(30, color=ORANGE, ls='--', lw=1.2)
ax.text(31, ax.get_ylim()[0] + 1, '30 shots', color=GRAY, fontsize=9)

# Label a few well-known players by StatsBomb player_id: (label, offset, alignment)
labels_to_show = {
    5246: ('Suárez', (-8, 4), 'right'),
    3672: ('Ibrahimović', (6, -3), 'left'),
    5497: ('Higuaín', (6, -3), 'left'),
    6399: ('Bale', (6, -3), 'left'),
    6973: ('Džeko', (6, -3), 'left'),
}
for _, row in players[players['player_id'].isin(labels_to_show)].iterrows():
    text, offset, align = labels_to_show[row['player_id']]
    ax.annotate(text, (row['shots'], row['goals_minus_xg']), xytext=offset,
                textcoords='offset points', fontsize=8.5, color='#0b0b0b', ha=align)

ax.set_xlabel('Non-penalty shots in 2015/16')
ax.set_ylabel('Goals minus StatsBomb xG')

fig.tight_layout()
fig.savefig(out_folder / 'fig4_players.png', dpi=200)
plt.close()

print('Saved figures to', out_folder.resolve())
