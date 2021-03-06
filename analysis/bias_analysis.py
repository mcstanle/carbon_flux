"""
This script should be able to do the following
1. Perform and output an across-time bias analysis
2. Perform and output

Author        : Mike Stanley
Created       : April 14, 2020
Last Modified : April 22, 2020

===============================================================================
- USE
I want to be able to pass
1. location of scale factor files of interest (this should be same directory
   where all output will go)

Write data and plots to directory structure

<directory for analysis>
  |
  '- data
    '- scale_factors
  '- monthly visuals
    '- #####
  '- total_time_visuals

===============================================================================
"""

import argparse
from glob import glob
import numpy as np
import PseudoNetCDF as pnc
import os

# plotting
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('ggplot')


def read_sf_objs(base_df_dir):
    """
    Reads in all objects present in the ./scale_factors directory

    Parameters:
        base_df_dir (str) : base directory where all scale factors can be found

    Returns:
        list of sf objects

    NOTE:
    - tracerinfo and diaginfo files must be present in the given directory
    """
    # obtain the scale factor file names (NOTE: file order doesn't matter)
    file_names = glob(base_df_dir + '/data/scale_factors/sf*')

    return [pnc.pncopen(fn, format='bpch') for fn in file_names]


def create_sf_arr(list_of_sf_objs, var_oi='IJ-EMS-$_CO2bal'):
    """
    Creates a 4D stacked array all scale factors across all OSSEs
    and months.

    Parameters:
        list_of_sf_objs (list) : list of pnc objects
        var_oi          (str)  : the variable of interest in each of the above
                                 elements

    Returns:
        - numpy array (# iterations, lon, lat)
        - longitude array
        - latitude array

    NOTE:
    - this function does not yet have the capability to handle multiple
      months of scale factors TODO
    """
    # extract the scale factors from each object
    extr_arrs = [sf_i.variables[var_oi].array()[0, :, :, :]
                 for sf_i in list_of_sf_objs]

    # stack the above
    stacked_arrs = np.stack(extr_arrs, axis=0)

    # obtain longitude and latitude
    lon = list_of_sf_objs[0].variables['longitude'].array()
    lat = list_of_sf_objs[0].variables['latitude'].array()

    return stacked_arrs, lon, lat


def find_bias(sf_stack, opt_sf):
    """
    Finds difference between avg(sf) and optimal sf for one 46x72 grid

    Parameters:
        sf_stack (numpy array) : nxMx46x72 array with inverted scale factors
                                 (n=number OSSEs and M=number of months)
        opt_sf   (numpy array) : Mx46x72 optimal scale factor array

    Returns:
        Mx46x76 numpy array of E(sf) - opt_sf

    NOTE:
     - we assume that the 0th index of sf_stack is the OSSE iterations
    """
    assert sf_stack.shape[1] == opt_sf.shape[0]

    # make sure the dimensions are correct
    if opt_sf.shape[1] != 46:

        # the only problem equiped to handle is lat/lon switch
        opt_sf_proc = np.swapaxes(opt_sf, axis1=1, axis2=2)

    else:
        opt_sf_proc = opt_sf.copy()

    # find mean of the given stack of sf draws
    sf_stack_avg = sf_stack.mean(axis=0)

    return sf_stack_avg - opt_sf_proc


def bias_map(bias_arr, lon, lat, title=None, write_loc=None):
    """
    Creates a geographic map of biases given a 46x72 bias array

    Parameters:
        bias_arr  (numpy arr) : 46x72 (latXlon)
        lon       (numpy arr) : longitude array
        lat       (numpy arr) : latitude array
        title     (str)       : (hopefully...) descriptive title
        write_loc (str)       : file's output location

    Returns:
        Nothing -- prints the map to file if write location
    """
    assert bias_arr.shape == (46, 72)

    fig = plt.figure(figsize=(12.5, 6))
    norm = colors.DivergingNorm(vcenter=0)

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
    contour = ax.contourf(
        lon, lat, bias_arr, levels=100,
        transform=ccrs.PlateCarree(), cmap='bwr', norm=norm
    )
    fig.colorbar(contour, ax=ax, orientation='vertical', extend='both')
    ax.add_feature(cfeature.COASTLINE)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if write_loc:
        plt.savefig(write_loc)
    else:
        plt.show()


def bias_map_total(bias_arr, lon, lat, title=None, write_loc=None):
    """
    Creates a geographic map of biases given a 46x72 bias array

    Parameters:
        bias_arr  (numpy arr) : 46x72 (latXlon)
        lon       (numpy arr) : longitude array
        lat       (numpy arr) : latitude array
        title     (str)       : (hopefully...) descriptive title
        write_loc (str)       : plot name - if None provides stem

    Returns:
        Nothing -- prints the map to file

    NOTE:
    - expects to receive a bias array of form (M, 46, 72)
    """
    # check the form of the bias arr
    assert bias_arr.shape[1] == 46
    assert bias_arr.shape[2] == 72

    # find average bias across all months
    sf_bias_tot = bias_arr.mean(axis=0)

    bias_map(
        bias_arr=sf_bias_tot,
        lon=lon,
        lat=lat,
        title=title,
        write_loc=write_loc
    )


def bias_map_monthly(bias_arr, lon, lat, write_dir):
    """
    Creates a geographic map of biases given a 46x72 bias array

    Parameters:
        bias_arr  (numpy arr) : 46x72 (latXlon)
        lon       (numpy arr) : longitude array
        lat       (numpy arr) : latitude array
        write_dir (str)       : directory where to write plots

    Returns:
        Nothing -- prints the map to file

    NOTE:
    - expects to receive a bias array of form (M, 46, 72)
    - There is no title capability for these plots
    """
    # check the form of the bias arr
    assert bias_arr.shape[1] == 46
    assert bias_arr.shape[2] == 72

    # check the the output directory exists
    assert os.path.isdir(write_dir)

    # identify the number of months
    NUM_MONTHS = bias_arr.shape[0]

    # create a plot for each month
    for month_idx in range(NUM_MONTHS):

        # create the file path
        file_path = write_dir + '/bias_month_%s' % str(month_idx).zfill(2)

        bias_map(
            bias_arr=bias_arr[month_idx, :, :],
            lon=lon,
            lat=lat,
            write_loc=file_path
        )


if __name__ == '__main__':

    # default values
    BASE_DIR = '/Users/mikestanley/Research/Carbon_Flux'
    RUN_NM = 'JULES_truth_CT_prior_agg_sf_files'
    SF_BASE_DIR = BASE_DIR + '/data/bias_calc_opt_output/' + RUN_NM
    OPT_SF_PATH = BASE_DIR + \
        '/data/optimal_scale_factors/2010_JULES_true_CT_prior/jan_sept.npy'

    # initialize the argparser
    parser = argparse.ArgumentParser()

    # fundamental arguments
    parser.add_argument('--sf_dir',
                        default=SF_BASE_DIR, type=str)
    parser.add_argument('--opt_df_dir',
                        default=OPT_SF_PATH, type=str)

    # parse the given arguments
    args = parser.parse_args()

    # read in all scale factor obj files
    print('Reading in raw SF files')
    sf_obj = read_sf_objs(base_df_dir=args.sf_dir)

    # create the numpy arrays of the scale factors
    print('Creating SF numpy arrays')
    sf_arr, lon, lat = create_sf_arr(list_of_sf_objs=sf_obj)
    print(sf_arr.shape)

    # read in the optimal scale factors
    print('Optimal scale factors read in')
    opt_sf = np.load(file=args.opt_df_dir)
    print(opt_sf.shape)

    # find the bias
    print('Bias obtained')
    sf_bias = find_bias(sf_stack=sf_arr, opt_sf=opt_sf)
    print(sf_bias.shape)

    # write the bias array
    print('Bias file saved')
    np.save(file=args.sf_dir + '/data/bias_arr.npy', arr=sf_bias)

    # check to see if map location is available
    map_dir = args.sf_dir + '/visualization'
    month_map_dir = map_dir + '/monthly_plots'

    if not os.path.isdir(map_dir):
        os.mkdir(map_dir)

    if not os.path.isdir(month_map_dir):
        os.mkdir(month_map_dir)

    # map over all months
    map_file_nm = map_dir + '/bias_map_all_months.png'
    print('Writing total time bias map to %s' % map_file_nm)
    bias_map_total(
        bias_arr=sf_bias,
        lon=lon,
        lat=lat,
        title='Scale Factor Bias 2010 Jan-Sept relative to Optimal Scale Factors',
        write_loc=map_file_nm
    )

    # map over individual months
    bias_map_monthly(
        bias_arr=sf_bias,
        lon=lon,
        lat=lat,
        write_dir=month_map_dir
    )
