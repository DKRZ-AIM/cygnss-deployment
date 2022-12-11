import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, colors, ticker
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import itertools
plt.switch_backend('agg')

deg = 1 # grid resolution (publication: 1)

grid_lon = np.arange(-180, 181, deg)
grid_lat = np.arange(-90, 91, deg)

def average_to_grid2(lon, lat, var, resolution=1, fill_value=-1):
    '''
    Grid a time-dependent variable in lon/lat and average over all counts
    
    lon - time series of lon coordinate (1D) (0...360)
    lat - time series of lat coordinate (1D)
    var - time series of variable (1D)
    resolution - target grid resolution (default: 1 deg)
    fill_value - a value that can be used for filling (i.e. that does not show up in var)
    
    Returns:
    2D gridded arrays for lat, lon, count-averaged var
    '''

    assert len(lon) == len(lat)
    assert len(lon) == len(var)

    grid_lon = np.arange(0, 360+resolution, resolution)
    grid_lat = np.arange(-90, 90+resolution, resolution)[::-1] # top left is +lat

    ix_lon = np.digitize(lon, grid_lon)
    ix_lat = np.digitize(lat, grid_lat)

    xx, yy = np.meshgrid(grid_lon, grid_lat, indexing='ij')
    gridded_var = np.empty(xx.shape, dtype='float')
    gridded_var[:] = fill_value

    ij = itertools.product(np.unique(ix_lon), np.unique(ix_lat))

    for i,j in ij:
        cond = (ix_lon==i) & (ix_lat==j)
        gridded_var[i,j] = np.mean(var[cond])

    gridded_var[gridded_var==fill_value] = None


    return xx, yy, gridded_var

def make_scatterplot(y_true, y_pred, date_):
    ymin = 2.5
    ymax = 25.0

    fig=plt.figure()
    ax=fig.add_subplot(111)

    img=ax.hexbin(y_true, y_pred, cmap='viridis', norm=colors.LogNorm(vmin=1, vmax=25000), mincnt=1)
    clb=plt.colorbar(img)
    clb.set_ticks([1, 10, 100, 1000, 10000])
    clb.set_ticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$'])
    clb.set_label('Samples in bin')
    clb.ax.tick_params()

    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('Predicted wind speed (m/s)')

    ax.plot(np.linspace(0, 30), np.linspace(0, 30), 'w:')

    ax.set_ylim(ymin, 25)
    ax.set_xlim(ymin, 25)

    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    ax.set_yticks([5, 10, 15, 20, 25])
    ax.set_yticklabels([5, 10, 15, 20, 25])

    fig.tight_layout()
    plt.savefig(f'/app/plots/scatter_{date_}.png')
     

def make_histogram(y_true, y_pred, date_):
    fig=plt.figure()
    ax=fig.add_subplot(111)

    sns.histplot(y_true, ax=ax, color='C7', label='ERA5 wind speed (m/s)')
    sns.histplot(y_pred, ax=ax, color='C2', label='Predicted wind speed (m/s)')

    ax.legend(fontsize=12)

    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    ax.set_xlabel('ERA5 wind speed (m/s)')

    plt.savefig(f'/app/plots/histo_{date_}.png')

def era_average(y_true, sp_lon, sp_lat, date_):
    xx, yy, gridded_y_true = average_to_grid2(sp_lon[:], sp_lat[:], y_true[:], resolution=deg)
    proj = ccrs.PlateCarree(180)
    
    fig, ax = plt.subplots(1, 1, figsize=(6,4), gridspec_kw=dict(hspace=0.05, wspace=0.1), subplot_kw=dict(projection=proj))
    cmap = ax.contourf(grid_lon[:], grid_lat[::-1][:], gridded_y_true[:].T, levels=60, transform=proj, antialiased=False, cmap='magma')
    ax.coastlines()
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0, color='gray', alpha=0.5, linestyle=':')
    gl.top_labels = False
    gl.right_labels= False
    clb = plt.colorbar(cmap, ax=ax, orientation='horizontal', shrink=1, label='Average ERA5 wind speed (m/s)')

    clb.set_ticks(np.arange(2.5, 18, 2.5))
    clb.ax.tick_params(labelsize=8)

    gl.xlabel_style = {'size': 8, 'color': 'black'}
    gl.ylabel_style = {'size': 8, 'color': 'black'}

    plt.savefig(f'/app/plots/era_average_{date_}.png')

def rmse_average(y_true, y_pred, sp_lon, sp_lat):
    xx, yy, gridded_rmse = average_to_grid2(sp_lon[:], sp_lat[:], np.abs(y_pred[:] - y_true[:]), resolution=deg)
    proj = ccrs.PlateCarree(180)
    fig, ax = plt.subplots(1, 1, figsize=(6,4), gridspec_kw=dict(hspace=0.05, wspace=0.1), subplot_kw=dict(projection=proj))
    cmap = ax.contourf(grid_lon[:], grid_lat[::-1][:], gridded_rmse[:].T, levels=60, transform=proj, antialiased=False, cmap='viridis')
    ax.coastlines()
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0, color='gray', alpha=0.5, linestyle=':')
    gl.top_labels = False
    gl.right_labels= False
    clb = plt.colorbar(cmap, ax=ax, orientation='horizontal', shrink=1, label='Average RMSE (m/s)')

    clb.set_ticks(np.arange(0, np.nanmax(gridded_rmse)+1, 1.0))
    clb.ax.tick_params(labelsize=8)

    gl.xlabel_style = {'size': 8, 'color': 'black'}
    gl.ylabel_style = {'size': 8, 'color': 'black'}


def today_longrunavg(df_mockup, y_bins, date_):
    
    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(111)

    sns.barplot(data=df_mockup, x='bins', y='rmse', hue='time', ax=ax)
    ax.legend()

    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('RMSE (m/s)')

    ax.set_xticks(range(len(y_bins)))
    ax.set_xticklabels([f'< {yy} m/s' for yy in y_bins])

    plt.savefig(f'/app/plots/today_longrunavg_{date_}.png')

def today_longrunavg_bias(df_mockup, y_bins, date_):

    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(111)

    sns.barplot(data=df_mockup, x='bins', y='bias', hue='time', ax=ax)
    ax.legend()

    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('Bias (m/s)')

    ax.set_xticks(range(len(y_bins)))
    ax.set_xticklabels([f'< {yy} m/s' for yy in y_bins])
    
    plt.savefig(f'/app/plots/today_long_bias_{date_}.png')

def sample_counts(df_rmse, y_bins, date_):

    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(111)
    sns.barplot(data=df_rmse, x='bins', y='counts', ax=ax)
    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('Sample counts')

    ax.set_xticks(range(len(y_bins)))
    ax.set_xticklabels([f'< {yy} m/s' for yy in y_bins])

    plt.savefig(f'/app/plots/sample_counts_{date_}.png')

def rmse_bins_era(df_rmse, y_bins, date_):

    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(111)
    sns.barplot(data=df_rmse, x='bins', y='rmse', ax=ax)
    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('RMSE (m/s)')

    ax.set_xticks(range(len(y_bins)))
    ax.set_xticklabels([f'< {yy} m/s' for yy in y_bins])

    plt.savefig(f'/app/plots/rmse_bins_era_{date_}.png')

def bias_bins_era(df_rmse, y_bins, date_):

    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(111)
    sns.barplot(data=df_rmse, x='bins', y='bias', ax=ax)
    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('Bias (m/s)')

    ax.set_xticks(range(len(y_bins)))
    ax.set_xticklabels([f'< {yy} m/s' for yy in y_bins])
 
    plt.savefig(f'/app/plots/bias_bins_era_{date_}.png')

