"""
Script to generate a battery of evaluation results for a given inversion.

Author:        Mike Stanley
Created:       Dec 8, 2019
Last Modified: Dec 8, 2019
================================================================================
TODO
1. find a better way to save the prior and truth used...potentially a config
   file in cheyenne?
"""

import argparse
from glob import glob
import os
import PseudoNetCDF as pnc
from shutil import copyfile
import xbpch

# constants
BASE_DIR = '/Users/mikestanley/Research/Carbon_flux'
INV_BASE_DIR = BASE_DIR + '/gc_adj_runs'
INV_NM = 'increased_prior_uncertainty_100'
PRIOR_DIR = BASE_DIR + '/data/NEE_fluxes'
TRUTH_DIR = BASE_DIR + '/data/JULES'
FLUX_PREFIX = 'nep.geos.4x5.'
FLUX_YEAR = 2010


def collect_data(
    inversion_nm=INV_NM, prior_dir=PRIOR_DIR, truth_dir=TRUTH_DIR,
    inversion_dir=INV_BASE_DIR,
    year=FLUX_YEAR, flux_prefix=FLUX_PREFIX
):
    """
    Collects all necessary data to produce desired results. These include
        - true fluxes
        - prior fluxes
        - scaling factors

    Parameters:
        inversion_nm  (str) : name of inversion we're interested in
        inversion_dir (str) : parent directory where inversion results are
        prior_dir     (str) : location of the prior fluxes
        truth_dir     (str) : location of the true fluxes
        year          (int) : year of fluxes
        flux_prefix   (str) : prefix of the fluxes to be read in

    Returns:
        dictionary containing data {'truth':, 'prior':, 'sf':}

    NOTE:
    - data_dir assumes that ./../data/ is where the the prior and truth are
    - each of the values in the dictionary is averaged over the month to which
      it belongs
    """
    # we assume structure of the location of the inversion results
    invert_dir = '/'.join([inversion_dir, inversion_nm])

    # binary punch parameter files
    diag_path = truth_dir + '/diaginfo.dat'
    tracer_path = truth_dir + '/tracerinfo.dat'

    # ---- true fluxes ----

    # get all files
    true_file_names = sorted(
        list(glob(truth_dir + '/%s%i.*' % (flux_prefix, year)))
    )

    # read in all the true fluxes
    # true_fluxes = xbpch.open_mfbpchdataset(true_file_names,
    #                                        dask=True,
    #                                        tracerinfo_file=tracer_path,
    #                                        diaginfo_file=diag_path)

    # ---- prior fluxes ----
    # read in all the priors
    prior_file_names = sorted(
        list(glob(prior_dir + '/%s.*' % flux_prefix))
    )

    # read in all the prior fluxes
    # prior_fluxes = xbpch.open_mfbpchdataset(prior_file_names,
    #                                         dask=True,
    #                                         tracerinfo_file=tracer_path,
    #                                         diaginfo_file=diag_path)

    # ---- scale factors ----

    # check if

    sf_months = list(glob(invert_dir + '/OptData/*'))

    sf_dict = {}
    for month in sf_months:
        # copy the tracer and diag files in
        os.popen('cp %s %s' % (invert_dir + '/tracerinfo.dat',
                               month + '/tracerinfo.dat'))
        os.popen('cp %s %s' % (invert_dir + '/diaginfo.dat',
                               month + '/diaginfo.dat'))

        # get the last sf iteration
        all_sf_files = list(glob(month + '/gctm.sf.*'))
        highest_iter = max([int(i.split('.')[-1]) for i in all_sf_files])
        conv_sf_file = [i for i in all_sf_files
                        if int(i.split('.')[-1]) == highest_iter][0]

        # read in the sf file
        sf = pnc.pncopen(conv_sf_file, format='bpch')

        # add to the dictionary
        sf_dict[month.split('/')[-1]] = sf

        # remove the tracer and diag files
        os.remove(month + '/tracerinfo.dat')
        os.remove(month + '/diaginfo.dat')

    return sf_dict

    # read in the sf file
    # sf = pnc.pncopen(conv_sf_file, format='bpch')


def main(inversion_nm,
         prior_dir=PRIOR_DIR,
         truth_dir=TRUTH_DIR,
         year=FLUX_YEAR,
         error_file=True, error_plot=True, geo_plot=True,
         base_dir=INV_BASE_DIR
         ):
    """
    Get results for a given inversion

    Parameters:
        inversion_nm (str)  : name of the inversion for which we want results
        prior_dir    (str)  : location of prior
        truth_dir    (str)  : location of the truth
        error_file   (bool) : flag to save error file
        error_plot   (bool) : flag to save error plot
        geo_plot     (bool) : flag to save geographical plots (all months)
        base_dir     (str)  : base directory to which items are stored

    Returns:
        None - creates files in
    """
    # define the inversion directory
    inv_dir = '/'.join([base_dir, inversion_nm])

    # check if the directory exists
    if not os.path.exists(inv_dir):
        raise IOError("%s does not exist" % inv_dir)

    # collect data
    data_dict = collect_data(
        inv_dir=inv_dir, prior_dir=prior_dir, truth_dir=truth_dir
    )


if __name__ == "__main__":

    # set up the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion_name', '-i',
                        help='name of inversion', default='<NO NAME GIVEN>')
    args = parser.parse_args()

    # create results
    main(inversion_nm=args.inversion_name)
