"""
This script is intended to create an ocean mask given some fluxes.

An ocean mask is defined as those 72x46 grid locations where the average
January monthly flux is 0. This mask is given in the ravel numpy form to
simplify storage as a vector.

Provide the following inputs:
1. base path where fluxes are to be found
2. individual flux file prefix to use
3. includsive bounds on flux file numbers

RETURNS: npy file at given location

Author:        Mike Stanley
Created:       Jan 23, 2020
Last Modified: Jan 23, 2020
================================================================================
"""

import argparse
import os

import numpy as np
import xbpch


# some operational constants
DEFAULT_LB = 1
DEFAULT_UB = 31
DEFAULT_FLUX_PREFIX = 'nep.geos.4x5.'
CARBON_FLUX_DIR = '/Users/mikestanley/Research/Carbon_Flux'
DEFAULT_TRACER_FP = CARBON_FLUX_DIR + '/data/tracerinfo.dat'
DEFAULT_DIAG_FP = CARBON_FLUX_DIR + '/data/diaginfo.dat'
DEFAULT_FLUX_NM = 'CO2_SRCE_CO2bf'


def read_daily_flux(
    flux_fp,
    flux_prefix,
    lb,
    ub,
    tracerfile_path,
    diagfile_path
):
    """
    Reads in a sequence of daily fluxes.

    Parameters:
        flux_fp         (str) : file path to fluxes
        flux_prefix     (str) : prefix to each flux file of interest
        lb              (str) : inclusive lower bound of flux file number
        ub              (str) : inclusive upper bound of flux file number
        tracerfile_path (str) : path to tracerinfo.dat
        diagfile_path   (str) : path to diaginfo.dat

    Returns:
        xarray core dataset containing fluxes
    """
    assert isinstance(flux_fp, str)
    assert isinstance(flux_prefix, str)

    # create a list of files to read in
    file_suffs = [f'{i:03}' for i in range(lb, ub + 1)]

    # create a list of flux filepaths
    flux_fps = ['%s/%s%s' % (flux_fp, flux_prefix, suff)
                for suff in file_suffs]

    # check that the above files exist
    for flux_file in flux_fps:
        assert os.path.exists(flux_file)

    # read in the files
    fluxes = xbpch.open_mfbpchdataset(
        flux_fps,
        dask=True,
        tracerinfo_file=tracerfile_path,
        diaginfo_file=diagfile_path
    )

    return fluxes


def create_ocean_mask(fluxes, flux_nm):
    """
    Determine where oceans are on 46x72 latitude X longitude grid.

    Parameters:
        fluxes  (xarray core dataset) : xarray object returned from
                                        read_daily_flux
        flux_nm (str)                 : name of flux variable in the flux files

   Returns:
        ravel array of (72,), (46,) length numpy arrays with ocean indices in
        a 72x46 numpy array. This corresponds to the indices of the flattened
        global grid.


    This function does the following:
    1. read in fluxes
    2. average fluxes
    3. return 0s on averaged array
    """
    # average the fluxes of the given flux name
    fluxes_mean = fluxes[flux_nm].values.mean(axis=0)

    # check we have the correct dimension orientation
    if fluxes_mean.shape != (72, 46):
        fluxes_mean = fluxes_mean.T

    # find the tuple of indices
    ocean_idxs = np.where(fluxes_mean == 0)

    return np.ravel_multi_index(ocean_idxs, dims=(72, 46))


if __name__ == '__main__':

    # initialize the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--flux_fp',
                        help='path to flux files', default=None)
    parser.add_argument('--flux_prefix',
                        help='flux prefix, e.g. nep.geos.4x5.',
                        default=DEFAULT_FLUX_PREFIX)
    parser.add_argument('--lb',
                        help='lowerbound on flux file number',
                        default=DEFAULT_LB)
    parser.add_argument('--ub',
                        help='upperbound on flux file number',
                        default=DEFAULT_UB)
    parser.add_argument('--tracer_fp',
                        help='filepath to tracerinfo.dat',
                        default=DEFAULT_TRACER_FP)
    parser.add_argument('--diag_fp',
                        help='filepath to diaginfo.dat',
                        default=DEFAULT_DIAG_FP)
    parser.add_argument('--flux_nm',
                        help='name of flux in bpch files',
                        default=DEFAULT_FLUX_NM)
    parser.add_argument('--save_fp', help='save npy location',
                        default=None)

    # parse the arguments
    args = parser.parse_args()

    print('---- GIVEN PARAMETERS ----')
    print('flux_fp     : %s' % args.flux_fp)
    print('flux_predix : %s' % args.flux_prefix)
    print('lower bound : %s' % args.lb)
    print('upper bound : %s' % args.ub)
    print('tracerinfo  : %s' % args.tracer_fp)
    print('diaginfo    : %s' % args.diag_fp)
    print('Flux Name   : %s' % args.flux_nm)
    print('save fp     : %s' % args.save_fp)
    print('--------------------------')

    # read in the fluxes
    fluxes = read_daily_flux(
        flux_fp=args.flux_fp,
        flux_prefix=args.flux_prefix,
        lb=args.lb,
        ub=args.ub,
        tracerfile_path=args.tracer_fp,
        diagfile_path=args.diag_fp
    )

    # find the ocean mask
    ocean_mask = create_ocean_mask(fluxes=fluxes, flux_nm=args.flux_nm)

    # save the above
    assert args.save_fp[-3:] == 'npy'
    np.save(args.save_fp, ocean_mask)
    print('Ocean mask saved to %s' % args.save_fp)
