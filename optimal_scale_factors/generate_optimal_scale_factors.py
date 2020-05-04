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
Last Modified : May 4, 2020
"""

import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize, Bounds
import xbpch

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


def read_dir_fluxes(flux_dir, flux_prefix, tracer_path, diag_path):
    """
    Read in a directory of bpch flux files

    Parameters:
        flux_dir    (str) : directory containing flux files
        flux_prefix (str) : prefix of fluxes of interest
        tracer_path (str) : path to tracerinfo file
        diag_path   (str) : path to diaginfo file

    Returns:
        xarray object containing the fluxes
    """
    # get flux file paths
    flux_paths = get_flux_file_paths(base_dir=flux_dir, prefix=flux_prefix)
    assert len(flux_paths) > 0

    # read in the flux files
    fluxes = xbpch.open_mfbpchdataset(
        flux_paths,
        dask=True,
        tracerinfo_file=tracer_path,
        diaginfo_file=diag_path
    )

    return fluxes


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


def find_month_idxs(
    fluxes,
    month_list=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep']
):
    """
    For some sequence of months, find the indices in the flux xarray
    corresponding to that month.

    Parameters:
        fluxes     (xarray) : output from read_dir_fluxes
        month_list (list)   :
    """
    # find the month indices
    month_idxs = {month: None for month in month_list}

    for month_idx, month in enumerate(month_idxs.keys()):
        # find the start and end values
        start_val = month_idx + 1
        end_val = month_idx + 2

        # find the time indices
        month_idxs[month] = find_month(
            fluxes=fluxes,
            start=start_val,
            end=end_val
        )

    return month_idxs


def cost_func(input_sfs, true_flux_vecs, prior_flux_vecs):
    """
    Evaluate the cost function for scale factors and fluxes

    This cost function is for evaluating a single month

    Parameters:
        input_sfs       (numpy arr): (lon x lat) x 1 (i.e. [(72 x 46) = 3312] x 1)
        true_flux_vecs  (numpy arr): time x (lon x lat) (i.e. ~248 x 3312)
        prior_flux_vecs (numpy arr): time x (lon x lat) (i.e. ~248 x 3312)

    Returns:
        Real value of cost
    """
    # reshape the input_sfs vector
    input_sfs_rs = input_sfs.reshape((input_sfs.shape[0], 1))

    return np.sum(
        np.square((true_flux_vecs.T - input_sfs_rs * prior_flux_vecs.T))
    )


def plot_cost_evals(cost_evals, month_nm, save_loc):
    """
    Plot the cost function evalutions to given save locations
    """
    plt.figure(figsize=(12.5, 6))
    plt.plot(cost_evals)

    # labels
    plt.xlabel('Iteration')
    plt.ylabel('Cost Eval')
    plt.title('Optimal SF Covergence : %s' % month_nm)

    plt.tight_layout()
    plt.savefig(save_loc)


def find_month_opt_sfs(
    month_idx,
    true_fluxes,
    prior_fluxes,
    flux_variable,
    obj_func,
    land_mask,
    space_dim,
    constrain,
    opt_meth,
    max_fun=50000
):
    """
    For a given month of raw data, we find the optimal scale factors..

    Parameters:
        month_idx     (np arr) : time indices for month of interest
        true_fluxes   (xarray) :
        prior_fluxes  (xarray) :
        flux_variable (str)    :
        month_idxs    (dict)   : indices in the above arrays associated with
                                 each month
        obj_func      (func)   : objective function used to learn optimal
                                 scale factors
        land_mask     (np arr) : simply numpy arr that given flatten spatial
                                 indices of land areas on the grid
        space_dim     (int)    : latitude times longitude dimension for
                                 flattened arrays
        constrain     (bool)   : lower bound flag
        opt_meth      (str)    : optimization method to use with
                                 scipy.optim.minimize
        max_fun       (int)    : maximum number of function calls

    Returns:
        a tuple with
            1. the numpy array of optimal scale factors
            2. the scipy.optim.minimize object which contains diagnostic info
    """
    # create monthly data
    true_month = true_fluxes[flux_variable].values[month_idx, :, :]
    prior_month = prior_fluxes[flux_variable].values[month_idx, :, :]

    # find the number of time steps in the month
    month_ts = true_month.shape[0]

    # create versions of the january fluxes for just land
    true_month_land_vec = true_month.reshape(
        (month_ts, space_dim)
    ).copy()[:, land_mask]
    prior_month_land_vec = prior_month.reshape(
        (month_ts, space_dim)
    ).copy()[:, land_mask]

    # function to capture the converge information for the optimization
    def save_cost_func(xk):
        global cost_evals
        cost_evals.append(obj_func(
            input_sfs=xk,
            true_flux_vecs=true_month_land_vec,
            prior_flux_vecs=prior_month_land_vec
        ))

    # find the length of the land vector
    land_vec_len = len(land_mask)

    # lower bounds
    if constrain:
        lb = np.zeros(land_vec_len)
    else:
        lb = np.array([-np.inf] * land_vec_len)

    # upper bounds -- infinite
    ub = np.array([np.inf] * land_vec_len)

    # create bounds object
    opt_bounds = Bounds(lb, ub)

    # optimize
    opt_sf_month = minimize(
        fun=obj_func,
        x0=np.ones((land_vec_len, 1)),
        args=(true_month_land_vec, prior_month_land_vec),
        bounds=opt_bounds,
        options={'maxfun': max_fun},
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
    true_f_dir,
    prior_f_dir,
    true_prefix,
    prior_prefix,
    tracer_path,
    diag_path,
    flux_variable,
    land_idx_fp,
    opt_method,
    constrain,
    opt_sf_save
):
    """
    Run the full optimal flux code

    Parameters:
        true_f_dir    (str)  :
        prior_f_dir   (str)  :
        true_prefix   (str)  :
        prior_prefix  (str)  :
        tracer_path   (str)  : path to tracerinfo file
        diag_path     (str)  : path to diaginfo file
        flux_variable (str)  : xarray variable name of interest
        land_idx_fp   (str)  : path to land flux array
        opt_method    (str)  : optimization method to use
        constrain     (bool) : flag to use lower bounded optimization
        opt_sf_save   (str)  : save dir of scale factors

    Returns:
        None - writes optimal scale factor array to file and saves a plot of
        convergence cost function
    """
    # read in true and prior fluxes
    true_fluxes = read_dir_fluxes(
        flux_dir=true_f_dir,
        flux_prefix=true_prefix,
        tracer_path=tracer_path,
        diag_path=diag_path
    )
    print('True Fluxes acquired')

    prior_fluxes = read_dir_fluxes(
        flux_dir=prior_f_dir,
        flux_prefix=prior_prefix,
        tracer_path=tracer_path,
        diag_path=diag_path
    )
    print('Prior Fluxes acquired')

    # find month indices
    month_idxs = find_month_idxs(fluxes=true_fluxes)
    print('Month indices determined')

    # determine some dimension constants
    flux_dims = true_fluxes[flux_variable].shape
    SPACE_DIM = flux_dims[1] * flux_dims[2]

    # get land indices
    ocean_mask = np.load(land_idx_fp)
    land_idx = np.setdiff1d(np.arange(0, SPACE_DIM), ocean_mask)

    # run the optimization
    optimal_sf_arrs = []
    optimize_output = {}
    cost_storage = []

    for month in tqdm(month_idxs.keys()):

        # find optimal scale factors for the month
        cost_evals = []
        pt_sf_month_full, opt_sf_month = find_month_opt_sfs(
            month_idx=month_idxs[month],
            true_fluxes=true_fluxes,
            prior_fluxes=prior_fluxes,
            flux_variable=flux_variable,
            obj_func=cost_func,
            land_mask=land_idx,
            space_dim=SPACE_DIM,
            constrain=constrain,
            opt_meth=opt_method,
            max_fun=50000
        )

        # save data
        optimal_sf_arrs.append(pt_sf_month_full)
        optimize_output[month] = opt_sf_month
        cost_storage.append(cost_evals)

        # make plot of convergence
        plot_cost_evals(
            cost_evals=cost_evals,
            month_nm=month,
            save_loc=opt_sf_save + '/%s_2010_CT_JULES.png' % month
        )


if __name__ == '__main__':

    # default values
    BASE_DIR = '/Users/mikestanley/Research/Carbon_Flux'
    TRACERINFO_PATH = BASE_DIR + '/data/JULES/tracerinfo.dat'
    DIAGINFO_PATH = BASE_DIR + '/data/JULES/diaginfo.dat'

    TRUE_F_LOC = BASE_DIR + '/data/JULES'
    PRIOR_F_LOC = BASE_DIR + '/NEE_fluxes'
    PREFIX_TRUE = 'nep.geos.4x5.2010.'
    PREFIX_PRIOR = 'nep.geos.4x5.'
    FLUX_VARIABLE = 'CO2_SRCE_CO2bf'
    LAND_IDX_FP = BASE_DIR + '/data/ocean_masks/JULES_jan_mask.npy'

    # optimization defaults
    POS_CONSTRAIN = False
    OPT_METHOD = 'L-BFGS-B'

    OPT_SF_SAVE_LOC = BASE_DIR + \
        '/data/optimal_scale_factors/2010_JULES_true_CT_prior/jan_sept.npy'

    # initialize the argparser
    parser = argparse.ArgumentParser()

    # fundamental arguments
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

    # run the optimization
    run(
        true_f_dir=args.true_f_dir,
        prior_f_dir=args.prior_f_dir,
        true_prefix=args.prefix_true,
        prior_prefix=args.prefix_prior,
        tracer_path=args.tracer_path,
        diag_path=args.diag_path,
        flux_variable=args.flux_variable,
        land_idx_fp=args.land_idx_fp,
        opt_method=args.opt_method
    )
