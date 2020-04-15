"""
When comparing OSSEs running on different machines, or with different
algorithm parameters (e.g. different number of iterations) it is helpful to
have a way to compare the resulting scale factors. This code is a way to
easily facilitate such a comparison. Namely, it does the following:

1. Plots of cost function from both OSSEs to confirm convergence.
2. Monthly map plots of scale factors and differences for direct comparison.
3. Scatter plot of optimized scale factors about the y=x line

==== INPUTS ====
1. directory paths to directories containing scale factor and cost function
   files (note these directories must contain the tracerinfo and diaginfo)
    - osse{1,2}_path
2. Plot output location

==== ASSUMPTIONS ====
1. the given scale factor files have the same number of months

Author        : Mike Stanley
Created       : Feb 5, 2020
Last Modified : Feb 6, 2020

==== USAGE ====
Look at ./scale_factor_comp.sh for a sample of how to use call this script.
"""

import argparse
from glob import glob

# dealing with data
import numpy as np
import PseudoNetCDF as pnc

# base plotting
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl
from matplotlib import colors

# plotting maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# for fitting a least squares line to scale factor plot
from sklearn.linear_model import LinearRegression

# constants
DEFAULT_FLUX_VAR = 'IJ-EMS-$_CO2bal'
DEFAULT_OUTPUT_PATH = '/Users/mikestanley/Research/Carbon_Flux'


def get_sf_data(osse_path, flux_var):
    """
    Gets last gctm.sf.NN file from given path

    Parameters:
        osse_path (str) : directory path where sf files are located
        flux_var  (str) : name of flux variable to consider

    Returns:
        tuple of the following numpy arrays
            - latitude
            - longitude
            - scale factor array with indices [month, lat, lon]

    """
    # find the last scale factor iteration file
    sf_fp = sorted(glob(osse_path + '/gctm.sf*'),
                   key=lambda x: int(x[-2:])
                   )[-1]

    # acquire scale factor pseudo netcdf file
    sf = pnc.pncopen(sf_fp)

    # get latitude and longitude
    lat = sf.variables['latitude'].array()
    lon = sf.variables['longitude'].array()

    # get the scale factors
    sf_arr = sf.variables[flux_var].array()[0, :, :, :]

    return lat, lon, sf_arr


def global_plots(
    sf1, sf2, lat, lon,
    output_loc, osse1_name, osse2_name, month_num
):
    """
    Create 3-part global plot to compare monthly scale factors
    1. OSSE1 scale factors
    2. OSSE2 scale factors
    3. OSSE1 - OSSE2

    Parameters:
        sf1  (numpy arr) : scale factors for OSSE1
        sf2  (numpy arr) : scale factors for OSSE2
        lat  (numpy arr) : numpy array of latitudes
        lon  (numpy arr) : numpy array of longitudes
        output_loc (str) : output directory
        osse1_name (str) : name of first OSSE
        osse2_name (str) : name of second OSSE
        month_num  (int) : number of month being optimized (jan == 0)

    Returns:
        None -- writes plots to output_loc

    NOTE: assumes that the output location contains all id info, like month
    """
    assert isinstance(month_num, int)

    # find the element-wise distance
    sf_diff = sf1 - sf2

    # get abs max to make all of the color bars comparable
    VMAX = np.max([
        np.abs(sf1).max(), np.abs(sf2).max(), np.abs(sf_diff).max()
    ])
    VMIN = -VMAX

    # ------- PLOTTING -------
    fig = plt.figure(figsize=(12.5, 14))

    # norms
    norm_sf = colors.DivergingNorm(vcenter=1, vmin=VMIN, vmax=VMAX)
    norm_diff = colors.DivergingNorm(vcenter=0, vmin=VMIN, vmax=VMAX)

    # plotting datasets
    sf1_plot = sf1.copy()
    sf2_plot = sf2.copy()
    diff_plot = sf_diff.copy()

    # add some min/max points to make color bars the same
    sf1_plot[-1, -1] = VMAX
    sf1_plot[-1, -2] = VMIN
    sf2_plot[-1, -1] = VMAX
    sf2_plot[-1, -2] = VMIN
    diff_plot[-1, -1] = VMAX
    diff_plot[-1, -2] = VMIN

    # sf1 map plot
    ax_sf1 = fig.add_subplot(311, projection=ccrs.PlateCarree(), aspect='auto')
    contour_sf1 = ax_sf1.contourf(
        lon, lat, sf1_plot, levels=100,
        transform=ccrs.PlateCarree(), cmap='bwr', norm=norm_sf
    )
    fig.colorbar(contour_sf1, ax=ax_sf1, orientation='vertical', extend='both')
    ax_sf1.add_feature(cfeature.COASTLINE)

    ax_sf1.set_title('%s SF (SF colors centered at 1)' % osse1_name)

    # Mike's Jan SFs
    ax_sf2 = fig.add_subplot(312, projection=ccrs.PlateCarree(), aspect='auto')
    contour_sf2 = ax_sf2.contourf(
        lon, lat, sf2_plot, levels=100,
        transform=ccrs.PlateCarree(), cmap='bwr', norm=norm_sf
    )
    fig.colorbar(contour_sf2, ax=ax_sf2, orientation='vertical', extend='both')
    ax_sf2.add_feature(cfeature.COASTLINE)

    ax_sf2.set_title('%s SF (SF colors centered at 1)' % osse2_name)

    # Differences -  Jan SFs
    ax_diff = fig.add_subplot(313, projection=ccrs.PlateCarree(),
                              aspect='auto')
    contour_diff = ax_diff.contourf(
        lon, lat, diff_plot, levels=100,
        transform=ccrs.PlateCarree(), cmap='bwr', norm=norm_diff
    )
    fig.colorbar(contour_diff, ax=ax_diff, orientation='vertical',
                 extend='both')
    ax_diff.add_feature(cfeature.COASTLINE)

    ax_diff.set_title('%s SF - %s SF' % (osse1_name, osse2_name))
    plt.suptitle('Comparison of Month %i' % month_num, fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

    save_loc = output_loc + '/global_comp_s%s_vs_%s_month%i.png' % (
        osse1_name, osse2_name, month_num
    )
    plt.savefig(save_loc)
    print('global plot saved to: %s' % save_loc)


def linear_comp_plots(sf1, sf2, output_loc, osse1_name, osse2_name, month_num):
    """
    Generate scatter plots of two OSSE scale factors

    Parameters:
        sf1  (numpy arr) : scale factors for OSSE1
        sf2  (numpy arr) : scale factors for OSSE2
        output_loc (str) : output directory
        osse1_name (str) : name of first OSSE
        osse2_name (str) : name of second OSSE
        month_num  (int) : number of month being optimized (jan == 0)

    Returns:
        None -- writes plots to output_loc
    """
    # create data and fit regression line
    num_rows = len(sf2.flatten())
    X = sf2.flatten().reshape((num_rows, 1))
    y = sf1.flatten()

    # max x and y value for plotting
    maxX = np.max(X)
    maxy = np.max(y)

    # fit a regression line to the above data
    reg = LinearRegression().fit(X, y)

    # create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(sf2.flatten(), sf1.flatten())

    # labels
    plt.xlabel('%s Opt Scale Factors' % osse2_name)
    plt.ylabel('%s Opt Scale Factors' % osse1_name)

    # plot the reg line
    x_s = np.arange(0, maxX, 0.25)
    y_hat = reg.predict(x_s[:, np.newaxis])
    plt.plot(x_s, y_hat, color='gray', alpha=0.5,
             label='Least squares fit: slope = %.3f\nR^2 = %.3f' %
             (reg.coef_[0], reg.score(X, y)))

    # plot y=x
    plt.plot(x_s, x_s, color='gray', alpha=0.5, linestyle='--', label='y=x')

    # ticks
    plt.xticks(np.arange(0, maxX, 0.25))
    plt.yticks(np.arange(0, maxy, 0.25))
    plt.xlim(-0.05, maxX)
    plt.ylim(-0.05, maxy)

    plt.legend(loc='best')

    plt.title('%s Opt. SFs vs. %s Opt. SFs - Month %i' % (
        osse1_name, osse2_name, month_num
    ))

    plt.tight_layout()
    save_loc = output_loc + '/scatter_plot_s%s_vs_%s_month%i.png' % (
        osse1_name, osse2_name, month_num
    )
    plt.savefig(save_loc)
    print('scatter plot saved to: %s' % save_loc)


def get_cost_func_vals(cfn_path):
    """
    Gets all files of the form cfn.* in the given file directory

    Parameters:
        cfn_path (str) : directory where the cost function vals are located

    Returns:
        List of cost function values, one per iteration
    """
    # get file names for cost functions
    cfn_fp = sorted(glob(cfn_path + '/cfn.*'), key=lambda x: int(x[-2:]))

    cfn = []
    for i, fp in enumerate(cfn_fp):
        with open(fp, 'r') as f:
            cfn.append(float(f.readlines()[0].replace(
                ' %i ' % (i + 1), ''
            ).replace(' ', '').replace('\n', '')))

    return cfn


def cost_fn_plot(cfn_path, osse_name, output_loc):
    """
    Plots the cost function values over the iterations to ensure convergence.

    Parameters:
        cfn_path   (str) : location of cfn.* files
        osse_name  (str) : name of the optimized fluxes
        output_loc (str) : directory location where to write plot

    Returns:
        Puts a plot in output_loc
    """
    # read in the cost function data
    cfn_osse = get_cost_func_vals(cfn_path)

    # plot the above
    plt.figure(figsize=(12.5, 6))

    plt.plot(range(1, len(cfn_osse) + 1), cfn_osse)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Fnc Val')
    plt.title('%s Cost Function Values Over Iterations' % osse_name)

    plt.tight_layout()
    save_loc = output_loc + '/cost_fn_%s.png' % osse_name
    plt.savefig(save_loc)
    print('cost fn plt saved to: %s' % save_loc)


def comparison_exec(
    osse1_path, osse2_path,
    osse1_name, osse2_name,
    output_dir, flux_var
):
    """
    Execute the comparison between the two OSSEs

    Parameters:
        osse1_path (str) : path to the first OSSE in the comparison
        osse2_path (str) : path to the second OSSE in the comparison
        osse1_name (str) : name of first OSSE
        osse2_name (str) : name of second OSSE
        output_dir (str) : output directory location for plots
        flux_var   (str) : name of the flux variable being compared

    Returns:
        None -- writes plots to plots to output_dir
    """
    # assert not osse1_path
    # assert not osse2_path
    # assert not output_dir
    # assert not osse1_name
    # assert not osse2_name

    # read in OSSE data
    lat, lon, sf_arr1 = get_sf_data(osse1_path, flux_var)
    lat, lon, sf_arr2 = get_sf_data(osse2_path, flux_var)

    # number of months
    num_months = sf_arr1.shape[0] - 1

    # for each month...
    for month_idx in range(num_months):

        # generate global plot
        global_plots(
            sf1=sf_arr1[month_idx, :, :],
            sf2=sf_arr2[month_idx, :, :],
            lat=lat,
            lon=lon,
            output_loc=output_dir,
            osse1_name=osse1_name,
            osse2_name=osse2_name,
            month_num=month_idx
        )

        # generate scatter plot
        linear_comp_plots(
            sf1=sf_arr1[month_idx, :, :],
            sf2=sf_arr2[month_idx, :, :],
            output_loc=output_dir,
            osse1_name=osse1_name,
            osse2_name=osse2_name,
            month_num=month_idx
        )

    # plot cost function OSSE1
    cost_fn_plot(
        cfn_path=osse1_path,
        osse_name=osse1_name,
        output_loc=output_dir
    )

    # plot cost function OSSE2
    cost_fn_plot(
        cfn_path=osse2_path,
        osse_name=osse2_name,
        output_loc=output_dir
    )


if __name__ == '__main__':

    # initialize arg parser
    parser = argparse.ArgumentParser()

    # add usage arguments
    parser.add_argument('--osse1_path',
                        help='cost and sf results from osse1', default=None)
    parser.add_argument('--osse2_path',
                        help='cost and sf results from osse2', default=None)
    parser.add_argument('--flux_variable',
                        help='variable in optimized flux output to look at',
                        default=DEFAULT_FLUX_VAR)
    parser.add_argument('--result_dir',
                        help='comparison result directory',
                        default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--osse1_name',
                        help='name of first OSSE',
                        default=None)
    parser.add_argument('--osse2_name',
                        help='name of second OSSE',
                        default=None)

    # parse the arguments
    args = parser.parse_args()

    # generate comparison between the OSSEs
    comparison_exec(
        osse1_path=args.osse1_path,
        osse2_path=args.osse2_path,
        osse1_name=args.osse1_name,
        osse2_name=args.osse2_name,
        output_dir=args.result_dir,
        flux_var=args.flux_variable
    )
