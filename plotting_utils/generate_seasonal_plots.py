"""
Code that takes as input some form of aggregated seasonal data and generates
4 seasonal plots, DJF, MAM, JJA, and SON.

Author:        Mike Stanley
Created On:    Nov 6, 2019
Last Modified: Nov 29, 2019

At the moment, this code is designed to be used within a jupyter notebook. The
code will generate a plot within the notebook environement and write the code
to file if a path is given.
"""

# plotting packages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from datetime import datetime
import glob
import numpy as np
# import PseudoNetCDF as pnc
import xbpch

SEASON_DICT = {
    '12': 'DJF',
    '01': 'DJF',
    '02': 'DJF',
    '03': 'MAM',
    '04': 'MAM',
    '05': 'MAM',
    '06': 'JJA',
    '07': 'JJA',
    '08': 'JJA',
    '09': 'SON',
    '10': 'SON',
    '11': 'SON',
}


def create_season_indices(datetime_arr, season_dict=SEASON_DICT):
    """
    Creates a tuple of indices that split datetime_arr into its seasonal parts

    Parameters:
        datetime_arr (numpy arr) : array of np.datetime64 values
        season_dict  (dict)      : mapping from month str (##) to season

    Returns:
        tuple of numpy arrays containing indices corresponding to:
            - DFJ
            - MAM
            - JJA
            - SON

    NOTE:
    - this function currently assumes that there is only one year of dates
      given, and that year is from 01-01 to 12-31
    """
    # extract months from numpy arr and label as seasons
    months = np.datetime_as_string(datetime_arr, unit='M')
    months = np.array([season_dict[i.split('-')[1]] for i in months])

    # DFJ
    dfj_indxs = np.where(months == 'DJF')[0]

    # MAM
    mam_indx = np.where(months == 'MAM')[0]

    # JJA
    jja_indx = np.where(months == 'JJA')[0]

    # SON
    son_indx = np.where(months == 'SON')[0]

    return dfj_indxs, mam_indx, jja_indx, son_indx


def generate_seaonal_data(dir_path, start_year, file_base, parameter):
    """
    Creates a dictionary for each of the seasons --
        - DJF
        - MAM
        - JJA
        - SON

    Parameters:
        dir_path   (str) : path to directory containing files
        start_year (int) : starting year for the fluxes
        file_base  (str) : form of files of interest
        parameter  (str) : parameter of interest in the underlying binary
                           punch files

    Returns:
        dictionary with keys corresponding to each of the seasons above. Each
        value is a list of all the days in that season.
        Also, has keys for
            - time
            - latitude
            - longitude

    NOTE:
    - files are assumed to be of the from filebase + .###, e.g.
      nep.geos.4x5.2010.001
    - the tracerinfo and diaginfo files are assumed to be in dir_path
    - files are assumed to be split into 3hour increments
    """
    START = datetime.now()

    # get all file names
    file_names = glob.glob(dir_path + file_base + '*')

    # read in the files
    fluxes = xbpch.open_mfbpchdataset(
        paths=file_names,
        dask=True,
        tracerinfo_file=dir_path + 'tracerinfo.dat',
        diaginfo_file=dir_path + 'diaginfo.dat'
    )
    print('Read in fluxes from %s' % dir_path + file_base)
    print('Elapsed time: %i seconds' % (datetime.now() - START).seconds)

    # generate flux dates from the given starting year
    end_data = np.datetime64('%s-01-01' % (start_year + 1))
    flux_dates = []
    start_date = np.datetime64('%s-01-01' % start_year)
    current_date = start_date

    while current_date < end_data:
        flux_dates.append(current_date)
        current_date += np.timedelta64(3, 'h')

    flux_dates = np.array(flux_dates)

    # create seasonal indices
    djf_indx, mam_indx, jja_indx, son_indx = create_season_indices(flux_dates)

    # create seasonal arrays
    djf_arr = fluxes[parameter].values[djf_indx, :, :]
    mam_arr = fluxes[parameter].values[mam_indx, :, :]
    jja_arr = fluxes[parameter].values[jja_indx, :, :]
    son_arr = fluxes[parameter].values[son_indx, :, :]

    # get latitude and longitude
    lat_arr = fluxes.lat.values
    lon_arr = fluxes.lon.values

    return {
        'time': flux_dates,
        'lat': lat_arr,
        'lon': lon_arr,
        'djf': djf_arr,
        'mam': mam_arr,
        'jja': jja_arr,
        'son': son_arr
    }


def plot_seasons(
    djf_arr,
    mam_arr,
    jja_arr,
    son_arr,
    lon_arr,
    lat_arr,
    vmin,
    vmax,
    plot_title,
    save_path=None
):
    """
    Produces 4 stacked subplots for each of the seasons.

    Parameters:
        djf_arr    (numpy arr) : array of dec, jan, feb values
        mam_arr    (numpy arr) : array of mar, apr, may values
        jja_arr    (numpy arr) : array of jun, jul, aug values
        son_arr    (numpy arr) : array of sep, oct, nov values
        lon_arr    (numpy arr) : arry of longitudinal coordinates
        lat_arr    (numpy arr) : arry of latitudinal coordinates
        vmin       (float)     : global min over contour values
        vmax       (float)     : global max over contour values
        plot_title (str)       : suptitle for all subplots
        save_path  (str)       : default None, if not, filepath to save loc

    Returns:
        matplotlib plotting object
        output file to filesystem if save_path is not None

    """
    fig = plt.figure(figsize=(12.5, 16))

    # djf
    ax_djf = fig.add_subplot(411, projection=ccrs.PlateCarree(), aspect='auto')

    # create contour
    djf_cont = ax_djf.contourf(lon_arr, lat_arr, djf_arr.T,
                               transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax)
    fig.colorbar(djf_cont, ax=ax_djf, orientation='vertical')
    ax_djf.set_title('DJF')

    # add map features
    ax_djf.add_feature(cfeature.COASTLINE)

    # mam
    ax_mam = fig.add_subplot(412, projection=ccrs.PlateCarree(), aspect='auto')

    # create contour
    mam_cont = ax_mam.contourf(lon_arr, lat_arr, mam_arr.T,
                               transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax)
    fig.colorbar(mam_cont, ax=ax_mam, orientation='vertical')
    ax_mam.set_title('MAM')

    # add map features
    ax_mam.add_feature(cfeature.COASTLINE)

    # jja
    ax_jja = fig.add_subplot(413, projection=ccrs.PlateCarree(), aspect='auto')

    # create contour
    jja_cont = ax_jja.contourf(lon_arr, lat_arr, jja_arr.T,
                               transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax)
    fig.colorbar(jja_cont, ax=ax_jja, orientation='vertical')
    ax_jja.set_title('JJA')

    # add map features
    ax_jja.add_feature(cfeature.COASTLINE)

    # son
    ax_son = fig.add_subplot(414, projection=ccrs.PlateCarree(), aspect='auto')

    # create contour
    son_cont = ax_son.contourf(lon_arr, lat_arr, son_arr.T,
                               transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax)
    fig.colorbar(son_cont, ax=ax_son, orientation='vertical')

    ax_son.set_title('SON')

    # add map features
    ax_son.add_feature(cfeature.COASTLINE)

    fig.subplots_adjust(right=1)

    plt.suptitle(plot_title, fontsize=20, y=0.93)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

# GARBAGE CODE SECTION
# # adjust subplots to make room for new color bar
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.02, hspace=0.2)

# # add axis for color bar
# cb_ax = fig.add_axes([0.83, 0.1, 0.02, .8]) # (from left, from bottom, width, also seems to move up and down)

# cont_vals = np.sort(np.concatenate((
#     djf_cont.cvalues,
#     mam_cont.cvalues,
#     jja_cont.cvalues, 
#     son_cont.cvalues
# )))

# im = cb_ax.imshow(cont_vals[:,np.newaxis], cmap='viridis', vmin=cont_vals.min(), vmax=cont_vals.max())
# cbar = fig.colorbar(djf_cont, cax=cb_ax)
# cbar.set_ticks(np.linspace(VMIN, VMAX, num=10))

# # get max and min of the above plot
# VMIN = cont_vals.min()
# VMAX = cont_vals.max()

# cmap = plt.get_cmap('jet') # may need to add N value back in
# norm = mpl.colors.Normalize(vmin=VMIN,vmax=VMAX)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# fig.colorbar(sm, ax=cb_ax, orientation='vertical')
