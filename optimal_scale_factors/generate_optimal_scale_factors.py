"""
This script takes a true set of fluxes and a prior set of fluxes and finds the
optimal scale factors to minimize the error between the true fluxes and the
scaled prior fluxes.

Inputs:
- true fluxes loc (str)
- prior fluxes loc (str)
- tracer/diag-info.dat paths (str)
- output location (str)
- used postively constrained set up

Outputs:
- saves a numpy array to the given location
- plots the cost function for the optimization for convergence checking

Author        : Mike Stanley
Created       : May 4, 2020
Last Modified : June 5, 2020
"""

from area import area
import argparse
from glob import glob
import pickle
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize, Bounds
import xbpch

import carbonfluxtools.computation as ccomp
import carbonfluxtools.io as cio

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def get_flux_file_paths(base_dir, prefix):
    """
    Obtain a list of file paths given a base directory and a prefix.

    Parameters:
        base_dir (str) :
        prefix   (str) : prefix that all files of interest have

    Returns:
        List of file names

    NOTE:
    - we assume that the end of each file name has a number on which we
      can order the files in the list.
    """
    file_p_base = base_dir + '/' + prefix + '*'

    return sorted([file_nm for file_nm in glob(file_p_base)])


def find_month(fluxes, start, end):
    """
    Find the time indices in the flux files between some start and end index

    Parameters:
        fluxes (int) : xarray flux dataset
        start  (int) : start month number (1-11) - inclusive
        end    (int) : end month number (2 - 12)

    Returns:
        all time indices where date is greater than or equal to start and less
        end
    """
    assert start < end
    if start > 9:
        less = np.where(
            fluxes.time.values < np.datetime64('1985-%i-01' % end)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-%i-01' % start)
        )[0]

    elif end > 9:
        less = np.where(
            fluxes.time.values < np.datetime64('1985-%i-01' % end)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-0%i-01' % start)
        )[0]
    else:
        less = np.where(
            fluxes.time.values < np.datetime64('1985-0%i-01' % end)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-0%i-01' % start)
        )[0]

    # find the intersection between the above
    time_idxs = np.intersect1d(geq, less)

    return time_idxs


def plot_cost_evals(cost_vals, month_nm, save_loc):
    """
    Plot the cost function evalutions to given save locations

    Parameters:
        cost_vals (list) : numeric values of cost function evaluated at each
                           iteration
        month_nm  (str)  : name of month being optimized
        save_loc  (str)  : file path for generated image

    Returns:
        None - writes plot to save_loc

    """
    plt.figure(figsize=(12.5, 6))
    plt.plot(cost_vals)

    # labels
    plt.xlabel('Iteration')
    plt.ylabel('Cost Eval')
    plt.title('Optimal SF Covergence : %s' % month_nm)

    plt.tight_layout()
    plt.savefig(save_loc)


def plot_month_sfs(sf_arr, lon, lat, write_loc=None):
    """
    Plot a global heatmap of scale factors for a single month

    Parameters:
        sf_arr    (np arr) : {lon} x {lat} array
        lon       (np arr) :
        lat       (np arr) :
        write_loc (str)    : file path to which plot should be written

    Returns:
        None - but write plot to file
    """
    assert sf_arr.shape[0] == 72
    assert sf_arr.shape[1] == 46

    fig = plt.figure(figsize=(12.5, 6))
    norm = colors.DivergingNorm(vcenter=1)

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
    contour = ax.contourf(
        lon, lat, sf_arr.T, levels=100,
        transform=ccrs.PlateCarree(), cmap='bwr', norm=norm
    )
    fig.colorbar(contour, ax=ax, orientation='vertical', extend='both')
    ax.add_feature(cfeature.COASTLINE)

    plt.savefig(write_loc)


def cost_func(
    input_sfs,
    true_month,
    prior_month,
    land_idx,
    ocean_idx,
    space_dim,
    lon,
    lat
):
    """
    Evaluate the cost function for scale factors and fluxes

    The evaluation criterion is essentiall root mean squared error of between
    posterior flux and true flux

    This cost function is for evaluating a single month

    Note, this function is designed to receive scale factors that have been
    removed of their ocean components. These need to be added in to take
    advantage of the weighted area RMSE function

    Parameters:
        input_sfs    (numpy arr) : (lon x lat) x 1
                                   (i.e. [(72 x 46) = 3312] x 1)
        true_month   (numpy arr) : time x (lon x lat) (i.e. ~248 x 72 x 46)
        prior_month  (numpy arr) : time x (lon x lat) (i.e. ~248 x 72 x 46)
        land_idx     (numpy arr) : indices of land grid cells
        ocean_idx    (numpy arr) : indices of ocean grid cells
        space_dim    (int)       : number of global grid cells
        lon          (numpy arr) : longitude array
        lat          (numpy arr) : latitude array

    Returns:
        Real value of cost
    """
    # insert ocean values back in - (3312,)
    opt_sf_month_full = np.ones(space_dim)
    opt_sf_month_full[land_idx] = input_sfs

    # reshape to (72, 46)
    opt_sf_month_full = opt_sf_month_full.reshape((72, 46))

    # find the mean for each grid cell
    true_flux_avg = true_month.mean(axis=0)
    prior_flux_avg = prior_month.mean(axis=0)

    # create gridwise square error array - (72, 46)
    gw_sq_err = np.square(true_flux_avg - prior_flux_avg * opt_sf_month_full)

    # find the weighted average square error
    w_err = ccomp.w_avg_flux_month(
        flux_arr=gw_sq_err,
        ocean_idxs=ocean_idx,
        lon=lon,
        lat=lat
    )

    return np.sqrt(w_err)


def compute_land_weights(ocean_mask, lon, lat):
    """
    Given a sequence of indices that indicate where ocean grid cells are,
    computes an array of weights to be used in the cost function for the
    optimization.

    Parameters:
        ocean_mask (np arr) : sequence of flattened indices (from 72x46)
        lon        (np arr) : longitude vector
        lat        (np arr) : latitude vector

    Returns:
        np arr of weights, should sum to 1
    """
    ref_arr = np.arange(46*72).reshape((72, 46))
    land_areas = []

    for lat_idx in np.arange(len(lat)):
        for lon_idx in np.arange(len(lon)):
            if ref_arr[lon_idx, lat_idx] in ocean_mask:
                continue

            # get the lower right corner endpoints of the box
            lon_lrc = lon[lon_idx] - 2.5
            lat_lrc = lat[lat_idx] - 2

            # find area
            land_areas.append(area(ccomp.subgrid_rect_obj(lon_lrc, lat_lrc)))

    # make the above into a numpy array
    land_a_arr = np.array(land_areas)

    # find the weights based upon these values
    land_weights = land_a_arr / land_a_arr.sum()

    assert land_weights.sum() == 1.

    return land_weights


def cost_lite(sfs, prior, truth, weigh):
    """
    Essentially the same function as above, but off-loading work within the
    above function to other parts of the code.

    THIS FUNCTION AVERAGES OVER MONTH BEFORE FINDING ERROR

    Parameters:
        sfs   (np arr) : proposed scale factors (L,) where L is len of land arr
        prior (np arr) : flattened prior mean monthly flux, only land
        truth (np arr) : flattened true mean monthly flux, only land
        weigh (np arr) : weight vector giving weights by land area

    Returns:
        float - RMSE with the land area weights
    """
    # find the squared errors
    sq_err = np.square(truth - prior * sfs)

    return np.sqrt(np.dot(sq_err, weigh))


def cost_lite_pre_avg(sfs, prior, truth, weigh):
    """
    Essentially the same function as above, but off-loading work within the
    above function to other parts of the code.

    THIS FUNCTION FINDS ERROR BEFORE AVERAGING OVER TIME (unlike cost_lite)

    prior * sfs is a (T x {land elems}) X (land elems) operation and then we
    average over time, which is axis 0. Then we take a weighted average by area
    size and return the square root of this value.

    Parameters:
        sfs      (np arr) : proposed scale factors for land grid cells (72*46)
        prior    (np arr) : full prior mean monthly flux (Tx(72*46))
        truth    (np arr) : full true mean monthly flux (Tx(72*46))
        weigh    (np arr) : weight vector giving weights by land area

    Returns:
        float - RMSE with the land area weights
    """
    # find MSE for each grid cell over time --> 72x46 : mean --> 72*46
    mse = np.mean(np.square(truth - prior * sfs), axis=0)

    return np.sqrt(np.dot(mse, weigh))


def find_month_opt_sfs(
    month,
    month_idx,
    true_flux_arr,
    prior_flux_arr,
    lon,
    lat,
    obj_func,
    land_mask,
    ocean_mask,
    land_weights,
    space_dim,
    constrain,
    opt_meth,
    max_fun=50000
):
    """
    For a given month of raw data, we find the optimal scale factors..

    Parameters:
        month          (str)    : name of month being optimized - used
                                  for updating the cost_evals list
        month_idx      (np arr) : time indices for month of interest
        true_flux_arr  (np arr) : (T x 72 x 46)
        prior_flux_arr (xarray) : (T x 72 x 46)
        lon            (np arr) : longitude array
        lat            (np arr) : latitude array
        month_idxs     (dict)   : indices in the above arrays associated with
                                  each month
        obj_func       (func)   : objective function used to learn optimal
                                  scale factors
        land_weights   (np arr) : weights for weighted cost function
        land_mask      (np arr) : simply numpy arr that given flatten spatial
                                  indices of land areas on the grid
        ocean_mask     (np arr) : provides indices of ocean grid cells
        space_dim      (int)    : latitude times longitude dimension for
                                  flattened arrays
        constrain      (bool)   : lower bound flag
        opt_meth       (str)    : optimization method to use with
                                  scipy.optim.minimize
        max_fun        (int)    : maximum number of function calls

    Returns:
        a tuple with
            1. the numpy array of optimal scale factors
            2. the scipy.optim.minimize object which contains diagnostic info

    NOTE:
    - depending on the obj_func being used, arguments need to be modified
    """
    # create monthly data Tx72x46
    true_month = true_flux_arr[month_idx, :, :]
    prior_month = prior_flux_arr[month_idx, :, :]

    # find the length of the land vector - 72*46 = 3312
    land_vec_len = len(land_mask)

    # find the number of time elements in this month
    time_len = len(month_idx)

    # reshape the above to be {num time steps} x {number of land grid spots}
    true_month_flat = true_month.reshape(time_len, 72*46)[:, land_mask]
    prior_month_flat = prior_month.reshape(time_len, 72*46)[:, land_mask]

    # find the flattened land only truth and prior
    # prior_flat_land = prior_month.mean(axis=0).flatten()[land_mask]
    # truth_flat_land = true_month.mean(axis=0).flatten()[land_mask]

    # function to capture the converge information for the optimization
    def save_cost_func(xk):
        global cost_evals
        cost_evals[month].append(obj_func(
            sfs=xk,
            truth=true_month_flat,
            prior=prior_month_flat,
            weigh=land_weights
        ))

    # lower bounds
    if constrain:

        # lower bound
        lb = np.zeros(land_vec_len)

        # upper bounds -- infinite
        ub = np.array([np.inf] * land_vec_len)

        # create bounds object
        opt_bounds = Bounds(lb, ub)

    # create some options
    if opt_meth == 'L-BFGS-B':
        options = {'maxfun': max_fun}
    elif opt_meth == 'BFGS':
        options = {'maxiter': 100}

    # optimize
    opt_sf_month = minimize(
        fun=obj_func,
        x0=np.ones(land_vec_len),
        args=(
            prior_month_flat,
            true_month_flat,
            land_weights
        ),
        bounds=opt_bounds if constrain else None,
        options=options,
        method=opt_meth,
        callback=save_cost_func
    )

    # go back to a full map of scale factors
    opt_sf_month_full = np.ones(space_dim)
    opt_sf_month_full[land_mask] = opt_sf_month['x']

    # reshape
    opt_sf_month_full = opt_sf_month_full.reshape((72, 46))

    return (opt_sf_month_full, opt_sf_month)


def run(
    read_txt,
    true_f_dir,
    prior_f_dir,
    true_prefix,
    prior_prefix,
    month_list,
    tracer_path,
    diag_path,
    flux_variable,
    land_idx_fp,
    opt_method,
    constrain,
    opt_sf_save,
    lon_loc,
    lat_loc,
    month_idx_fp=None
):
    """
    Run the full optimal flux code

    Parameters:
        read_txt      (bool) : determines file IO and month indexing
        true_f_dir    (str)  :
        prior_f_dir   (str)  :
        true_prefix   (str)  :
        prior_prefix  (str)  :
        month_list    (list) : list of months over which to optimize
        tracer_path   (str)  : path to tracerinfo file
        diag_path     (str)  : path to diaginfo file
        flux_variable (str)  : xarray variable name of interest
        land_idx_fp   (str)  : path to land flux array
        opt_method    (str)  : optimization method to use
        constrain     (bool) : flag to use lower bounded optimization
        opt_sf_save   (str)  : save dir of scale factors
        lon_loc       (str)  : file location of longitude numpy array
        lat_loc       (str)  : file location of latitude numpy array
        month_idx_fp  (str)  : file path to month indices -- only use when
                               read_txt == True

    Returns:
        None - writes optimal scale factor array to file and saves a plot of
        convergence cost function
    """
    global cost_evals

    # read in the lat/lon arrs
    lon = np.load(lon_loc)
    lat = np.load(lat_loc)

    if not read_txt:   # ====== FILE IO with BPCH files ======

        # read in true and prior fluxes
        true_fluxes = cio.read_flux_files(
            file_dir=true_f_dir,
            file_pre=true_prefix,
            tracer_fp=tracer_path,
            diag_fp=diag_path
        )
        print('True Fluxes acquired')

        prior_fluxes = cio.read_flux_files(
            file_dir=prior_f_dir,
            file_pre=prior_prefix,
            tracer_fp=tracer_path,
            diag_fp=diag_path
        )
        print('Prior Fluxes acquired')

        # extract numpy arrays from the above
        true_full = true_fluxes[flux_variable].values
        prior_full = prior_fluxes[flux_variable].values

        # find month indices
        month_idxs = cio.find_month_idxs(
            fluxes=true_fluxes,
            month_list=month_list
        )
        print('Month indices determined')

    else:   # ====== FILE IO with TXT files ======

        assert month_idx_fp

        # obtain the files names
        true_flux_files = sorted(
            glob(true_f_dir + '/' + true_prefix + '[0-9][0-9][0-9]'),
            key=lambda x: int(x[-3:])
        )

        prior_flux_files = sorted(
            glob(prior_f_dir + '/' + prior_prefix + '[0-9][0-9][0-9]'),
            key=lambda x: int(x[-3:])
        )

        # read in the fluxes
        true_full = cio.read_flux_txt_files(flux_files=true_flux_files)
        prior_full = cio.read_flux_txt_files(flux_files=prior_flux_files)

        print('True flux arra shape: %s' % str(true_full.shape))
        print('Prior flux arra shape: %s' % str(prior_full.shape))
        assert true_full.shape[0] != 0    # very basic checks
        assert prior_full.shape[0] != 0

        # read month indices from pickle file
        with open(month_idx_fp, 'rb') as f:
            month_idxs = pickle.load(f)

    # determine some dimension constants
    flux_dims = true_full.shape
    SPACE_DIM = flux_dims[1] * flux_dims[2]

    # get land indices
    ocean_mask = np.load(land_idx_fp)
    land_idx = np.setdiff1d(np.arange(0, SPACE_DIM), ocean_mask)

    # compute land weights
    land_weights = compute_land_weights(
        ocean_mask=ocean_mask, lon=lon, lat=lat
    )

    # run the optimization
    optimal_sf_arrs = []
    optimize_output = {}

    print('Starting optimization')
    count = 0
    print(month_idxs.keys())
    for month in tqdm(month_idxs.keys()):

        # find optimal scale factors for the month
        pt_sf_month_full, opt_sf_month = find_month_opt_sfs(
            month=month,
            month_idx=month_idxs[month],
            true_flux_arr=true_full,
            prior_flux_arr=prior_full,
            lon=lon,
            lat=lat,
            obj_func=cost_lite_pre_avg,
            land_mask=land_idx,
            ocean_mask=ocean_mask,
            land_weights=land_weights,
            space_dim=SPACE_DIM,
            constrain=constrain,
            opt_meth=opt_method,
            max_fun=50000
        )

        # save data
        optimal_sf_arrs.append(pt_sf_month_full)
        optimize_output[month] = opt_sf_month

        # make plot of convergence
        plot_cost_evals(
            cost_vals=cost_evals[month],
            month_nm=month,
            save_loc=opt_sf_save + '/%s_%s_convergence.png' % (
                str(count).zfill(2), month
            )
        )
        count += 1

    # save the scale factors
    opt_sf = np.stack(optimal_sf_arrs)
    np.save(
        file=opt_sf_save + '/%s_%s_opt_sfs.npy' % (
            month_list[0], month_list[-1]
        ),
        arr=opt_sf
    )

    # create plots for each month
    for idx, month in tqdm(enumerate(month_idxs.keys())):
        plot_month_sfs(
            sf_arr=opt_sf[idx, :, :],
            lon=lon,
            lat=lat,
            write_loc=opt_sf_save + '/%s_%s_sf_plot.png' % (
                str(idx).zfill(2), month
            )
        )


if __name__ == '__main__':

    # default values
    READ_TXT = True
    BASE_DIR = '/Users/mikestanley/Research/Carbon_Flux'
    TRACERINFO_PATH = BASE_DIR + '/data/JULES/tracerinfo.dat'
    DIAGINFO_PATH = BASE_DIR + '/data/JULES/diaginfo.dat'

    TRUE_F_LOC = BASE_DIR + '/data/JULES_YEAR_txt'
    PRIOR_F_LOC = BASE_DIR + '/data/NEE_fluxes_txt_scl'
    PREFIX_TRUE = 'nep.geos.4x5.2010.'
    PREFIX_PRIOR = 'nep.geos.4x5.'
    FLUX_VARIABLE = 'CO2_SRCE_CO2bf'
    LAND_IDX_FP = BASE_DIR + '/data/ocean_masks/JULES_jan_mask.npy'

    # default list of months
    MONTH_LST = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep']

    # lon/lat arr locations for plotting
    LON_LOC = BASE_DIR + '/data/lon_lat_arrs/lon.npy'
    LAT_LOC = BASE_DIR + '/data/lon_lat_arrs/lat.npy'

    # month index path -- only relevant if READ_TXT == True
    MONTH_IDX_FP = BASE_DIR + '/data/month_idxs/jan_sept.pkl'

    # optimization defaults
    POS_CONSTRAIN = False
    if POS_CONSTRAIN:
        OPT_METHOD = 'L-BFGS-B'
    else:
        OPT_METHOD = 'BFGS'

    OPT_SF_SAVE_LOC = BASE_DIR + \
        '/data/optimal_scale_factors/2010_JULES_true_CT_scl_prior_nc'

    # initialize the argparser
    parser = argparse.ArgumentParser()

    # fundamental arguments
    parser.add_argument('--read_txt',
                        default=READ_TXT, type=bool)
    parser.add_argument('--base_dir',
                        default=BASE_DIR, type=str)
    parser.add_argument('--true_f_dir',
                        default=TRUE_F_LOC, type=str)
    parser.add_argument('--prior_f_dir',
                        default=PRIOR_F_LOC, type=str)
    parser.add_argument('--prefix_true',
                        default=PREFIX_TRUE, type=str)
    parser.add_argument('--prefix_prior',
                        default=PREFIX_PRIOR, type=str)
    parser.add_argument('--tracer_path',
                        default=TRACERINFO_PATH, type=str)
    parser.add_argument('--diag_path',
                        default=DIAGINFO_PATH, type=str)
    parser.add_argument('--pos_constrain',
                        default=POS_CONSTRAIN, type=bool)
    parser.add_argument('--opt_method',
                        default=OPT_METHOD, type=str)
    parser.add_argument('--flux_variable',
                        default=FLUX_VARIABLE, type=str)
    parser.add_argument('--land_idx_fp',
                        default=LAND_IDX_FP, type=str)
    parser.add_argument('--save_loc',
                        default=OPT_SF_SAVE_LOC, type=str)

    # parse the given arguments
    args = parser.parse_args()

    # define a place for the cost function evaluations to go
    cost_evals = {month: [] for month in MONTH_LST}

    # run the optimization
    run(
        read_txt=args.read_txt,
        true_f_dir=args.true_f_dir,
        prior_f_dir=args.prior_f_dir,
        true_prefix=args.prefix_true,
        prior_prefix=args.prefix_prior,
        month_list=MONTH_LST,
        tracer_path=args.tracer_path,
        diag_path=args.diag_path,
        flux_variable=args.flux_variable,
        land_idx_fp=args.land_idx_fp,
        opt_method=args.opt_method,
        constrain=args.pos_constrain,
        opt_sf_save=args.save_loc,
        lon_loc=LON_LOC,
        lat_loc=LAT_LOC,
        month_idx_fp=MONTH_IDX_FP
    )
