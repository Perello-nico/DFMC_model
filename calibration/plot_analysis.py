"""
@author: Nicolò Perello
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


###################################################################
# PLOT GoF Matrix
###################################################################
METRICS_PALETTE = {
    'MAE': 'positive',
    'BIAS': 'centered-0',
    'RMSE': 'positive',
    'NNSE': 'centered-0.5'
}

METRICS_UM = {
    'MAE': '%',
    'BIAS': '%',
    'RMSE': '%',
    'NNSE': 'adim'
}


def plot_gof(df_TS_gof: pd.DataFrame,
             Nrows_grid: int = 24, Ncols_grid: int = 17,
             labels: list = None):
    aspect = 20
    pad_fraction = 0.5
    Ncols = 2
    Nrows = int(len(METRICS_PALETTE)/Ncols)
    fig = plt.figure(figsize=(7.5*Ncols, 7.5*Nrows), dpi=100)
    gs = GridSpec(Nrows, Ncols, figure=fig)
    col = 0
    row = 0
    for mm in METRICS_PALETTE:
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f'Metric: {mm}', fontsize=13)
        col += 1
        if col % Ncols == 0:
            col = 0
            row += 1
        vmin = np.min(df_TS_gof[mm])
        vmax = np.max(df_TS_gof[mm])
        if 'centered' in METRICS_PALETTE[mm]:  # palette centered in zero
            center_value = float(METRICS_PALETTE[mm].split('-')[1])
            norm = mcolors.TwoSlopeNorm(vmin=vmin,
                                        vcenter=center_value,
                                        vmax=vmax)
            c = ax.pcolormesh(df_TS_gof[mm].values.reshape(Nrows_grid,
                                                           Ncols_grid),
                              cmap=cm.RdBu_r, norm=norm,
                              edgecolors='w')
        else:
            c = ax.pcolormesh(df_TS_gof[mm].values.reshape(Nrows_grid,
                                                           Ncols_grid),
                              cmap=cm.Reds, vmin=vmin, vmax=vmax,
                              edgecolors='w')

        # Plotting labels if provided
        if labels is not None:
            if len(labels) != Nrows_grid * Ncols_grid:
                raise ValueError('Wrong number of labels')
            for i in range(Nrows_grid):
                for j in range(Ncols_grid):
                    label = labels[i * Ncols_grid + j]
                    ax.text(j + 0.5, i + 0.5, f'{label}',
                            ha='center', va='center',
                            fontsize=8, color='black')

        ax.set(xticks=[], yticks=[])
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes('right', size=width, pad=pad)
        cbar = plt.colorbar(c, cax=cax, orientation='vertical')
        cbar.set_label(f'[{METRICS_UM[mm]}]', rotation=90)
        plt.box(True)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    return fig


###################################################################
# PLOT Matrix Time Series
###################################################################
def matrix_TS(df_TS: pd.DataFrame, df_TS_gof: pd.DataFrame,
              Nrows: int = 24, Ncols: int = 17):
    Nts = len(df_TS)
    all_idx_ts = np.arange(Nts)
    schema_idx = all_idx_ts.reshape(Nrows, Ncols)
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(Nrows, Ncols, figure=fig)
    for idx_ts in all_idx_ts:
        idx_grid = np.argwhere(schema_idx == idx_ts)[0]
        ax = fig.add_subplot(gs[Nrows-(idx_grid[0]+1), idx_grid[1]])
        ffmc_obs = df_TS.iloc[idx_ts]['DFMC'].values
        ffmc_mod = df_TS_gof.iloc[idx_ts]['DFMC_model'].values
        ax.plot(ffmc_obs, color='black', linewidth=0.8)
        ax.plot(ffmc_mod, color='tomato', linewidth=0.8)
        max_ = np.max([ffmc_obs.max(), ffmc_mod.max()])
        ax.set_yticks(np.arange(0, max_, 5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return fig


###################################################################
# PLOT Points
###################################################################
def plot_points(df_TS: pd.DataFrame, df_TS_gof: pd.DataFrame,
                clusters: dict = None):
    N_TS = len(df_TS)
    obs = pd.concat([df_TS.iloc[ts]['DFMC'] for ts in range(N_TS)],
                    axis=0).values
    mod = pd.concat([df_TS_gof.iloc[ts]['DFMC_model'] for ts in range(N_TS)],
                    axis=0).values
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    if not (clusters is None):
        Nrepeats = [len(df_TS.iloc[ts].DFMC.values) for ts in range(N_TS)]
        clusters_val = df_TS.cluster.values.repeat(Nrepeats)
        for cc, color in clusters.items():
            idx = clusters_val == cc
            ax.scatter(obs[idx], mod[idx],
                       c=color, s=0.8,
                       label=f'{cc}')
        ax.legend()
    else:
        ax.scatter(obs, mod, c='royalblue', s=0.8)
    range_plot = [0, 45]
    ax.set_xlabel('Fuel moisture observed [%]')
    ax.set_ylabel('Fuel moisture modeled [%]')
    ax.set_xlim(range_plot)
    ax.set_ylim(range_plot)
    ax.set_aspect('equal', adjustable='box')
    ax.plot(range_plot, range_plot, color='black', linestyle='--')
    return fig


###################################################################
# PLOT Time Series
###################################################################
def plot_time_series(time_series: pd.DataFrame):
    time_serie = time_series.replace(to_replace=-9999,
                                     value=np.nan)
    time_serie['time'] = pd.to_datetime(time_serie['time'].values)
    time_serie.loc[:, 'day_night'] = np.where((time_serie.time.dt.hour >= 6) &
                                              (time_serie.time.dt.hour <= 18),
                                              'gold', 'royalblue')
    color = np.full(len(time_serie.phase.values), 'w', dtype='object')
    color[time_serie.phase.values == 1] = 'b'
    color[time_serie.phase.values == -1] = 'r'
    color[time_serie.phase.values == 2] = 'grey'
    time_serie.loc[:, 'color_phase'] = color
    Nrows = 6
    Ncols = 1
    fig = plt.figure(figsize=(10*Ncols, 1.5*Nrows), dpi=100)
    gs = GridSpec(Nrows, Ncols, figure=fig)
    lines_lgd = []
    # DFMC
    ax = fig.add_subplot(gs[0, 0])
    time_serie['DFMC'].plot(ax=ax, label='DFMC - observed', color='k')
    lines_lgd.append(['k', '-', 'DFMC - observed'])
    time_serie['DFMC_model'].plot(ax=ax, label='DFMC - model', color='green')
    lines_lgd.append(['green', '-', 'DFMC - model'])
    time_serie['EMC'].plot(ax=ax, label='EMC', linestyle='--', color='green')
    lines_lgd.append(['green', '--', 'EMC'])
    for i in time_serie.index:
        ax.axvspan(i-.5, i+.5, ymin=0.95, ymax=1,
                   facecolor=time_serie.loc[i].day_night, alpha=1)
        ax.axvspan(i-.5, i+.5, ymin=0, ymax=0.95,
                   facecolor=time_serie.loc[i].color_phase, alpha=0.2)
    ax.set_ylabel('[%]')
    ax.set_xlabel('time [h]')
    ax.label_outer()
    ax.grid()
    ax.set_xlim(left=0, right=len(time_serie)-1)
    # K
    ax = fig.add_subplot(gs[1, 0])
    time_serie['K_const'].plot(ax=ax, label='Response time',
                               color='darkviolet')
    lines_lgd.append(['darkviolet', '-', 'Response time'])
    for i in time_serie.index:
        ax.axvspan(i-.5, i+.5, ymin=0.95, ymax=1,
                   facecolor=time_serie.loc[i].day_night, alpha=1)
        ax.axvspan(i-.5, i+.5, ymin=0, ymax=0.95,
                   facecolor=time_serie.loc[i].color_phase, alpha=0.2)
    ax.legend().set_visible(False)
    ax.set_ylabel('[h]')
    ax.set_xlabel('time [h]')
    ax.label_outer()
    ax.grid()
    ax.set_xlim(left=0, right=len(time_serie)-1)
    # RH
    ax = fig.add_subplot(gs[2, 0])
    time_serie['Hum'].plot(ax=ax, label='Relative humidity', color='royalblue')
    lines_lgd.append(['royalblue', '-', 'Relative humidity'])
    ax.legend().set_visible(False)
    for i in time_serie.index:
        ax.axvspan(i-.5, i+.5, ymin=0.95, ymax=1,
                   facecolor=time_serie.loc[i].day_night, alpha=1)
    ax.set_ylabel('[%]')
    ax.set_ylim(0, 100)
    ax.set_xlabel('time [h]')
    ax.label_outer()
    ax.grid()
    ax.set_xlim(left=0, right=len(time_serie)-1)
    # Temp
    ax = fig.add_subplot(gs[3, 0])
    time_serie['Temp'].plot(ax=ax, label='Temperature', color='tomato')
    lines_lgd.append(['tomato', '-', 'Temperature'])
    ax.legend().set_visible(False)
    for i in time_serie.index:
        ax.axvspan(i-.5, i+.5, ymin=0.95, ymax=1,
                   facecolor=time_serie.loc[i].day_night, alpha=1)
    ax.set_ylabel('[°C]')
    ax.set_xlabel('time [h]')
    ax.label_outer()
    ax.grid()
    ax.set_xlim(left=0, right=len(time_serie)-1)
    # Wind
    ax = fig.add_subplot(gs[4, 0])
    time_serie['Wspeed'].plot(ax=ax, label='Wind speed', color='darkblue')
    lines_lgd.append(['darkblue', '-', 'Wind speed'])
    ax.legend().set_visible(False)
    for i in time_serie.index:
        ax.axvspan(i-.5, i+.5, ymin=0.95, ymax=1,
                   facecolor=time_serie.loc[i].day_night, alpha=1)
    ax.set_ylabel('[m/s]')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('time [h]')
    ax.label_outer()
    ax.grid()
    ax.set_xlim(left=0, right=len(time_serie)-1)
    # rain
    ax = fig.add_subplot(gs[5, 0])
    time_serie['Rain'].plot(ax=ax, label='Rain', color='grey')
    lines_lgd.append(['grey', '-', 'Rain'])
    ax.legend().set_visible(False)
    for i in time_serie.index:
        ax.axvspan(i-.5, i+.5, ymin=0.95, ymax=1,
                   facecolor=time_serie.loc[i].day_night, alpha=1)
    ax.set_ylabel('[mm]')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('time [h]')
    ax.label_outer()
    ax.grid()
    ax.set_xlim(left=0, right=len(time_serie)-1)

    ax.set_xticks(np.arange(0, len(time_serie),
                            step=max(len(time_serie)//10, 1)))
    ax.set_xticklabels(time_serie['time'].dt.strftime(
        '%Y-%m-%d %H:%M').values[::max(len(time_serie)//10, 1)],
                       rotation=45, ha='right')

    handles, labels = add_legend_all(lines_lgd)
    fig.axes[0].legend(handles, labels, loc='lower left',
                       bbox_to_anchor=(1, -0.4))
    plt.subplots_adjust(hspace=0.1)
    return fig


def add_legend_all(lines_lgd):
    handles = []
    labels = []
    # LINES
    for ll in lines_lgd:
        handles.append(Line2D([0], [0], color=ll[0], linestyle=ll[1], lw=2))
        labels.append(ll[2])
    # DAY/NIGHT
    handles.append(Patch(facecolor='gold', edgecolor='white', alpha=1))
    labels.append('day')
    handles.append(Patch(facecolor='royalblue', edgecolor='white', alpha=1))
    labels.append('night')
    # PHASES
    handles.append(Patch(facecolor='red', edgecolor='white', alpha=0.2))
    labels.append('drying phase')
    handles.append(Patch(facecolor='blue', edgecolor='white', alpha=0.2))
    labels.append('wetting phase')
    handles.append(Patch(facecolor='grey', edgecolor='white', alpha=0.2))
    labels.append('rain phase')
    return handles, labels
