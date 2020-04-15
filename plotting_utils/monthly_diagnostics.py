"""
Generates the following diagnostic output from a month long inversion:
1. Plot of stacked truth/prior/posterior of mean fluxes across globe
2. Output file with quantitative diagnostics:
    - Prior MSE
    - Posterior MSE
    - ratio of posterior / prior

Author        : Mike Stanley
Created       : December 4, 2019
Last Modified : December 4, 2019
===============================================================================
NOTE:
- the tracerinfo.dat and diaginfo.dat files are assumed to be included with the
  flux directories
"""

import glob
import numpy as np
import PseudoNetCDF as pnc
import xbpch

# constants
FLUX_FIELD = 'CO2_SRCE_CO2bf'


def generate_flux_filenames(directory_path, file_prefix):
    """
    Create list of filenames that will be read in by get_fluxes

    Parameters:
        directory_path (str) : location of fluxes of interest
        file_prefix    (str) : prefix of desired flux files, e.g.
                               "nep.geos.4x5.2010."

    Returns:
        A list of file names that can be given directly to get_fluxes()
    """
    file_names = sorted(
        [file_nm for file_nm
         in glob.glob(directory_path + file_prefix + '*')]
    )

    return file_names


def get_mean_fluxes(
    directory_path, file_prefix, month, flux_field=FLUX_FIELD
):
    """
    Obtain mean fluxes for month of interest
        -- can be used for truth and prior.

    Parameters:
        directory_path (str) : see generate_flux_filenames docstring
        file_prefix    (str) : see generate_flux_filenames docstring
        month          (int) : integer representation of month of interest
        flux_field     (str) : name of flux field in flux files

    Returns:
        dictionary with following keys (all numpy arrays)
        - flux
        - latitude
        - longitude
        - time
    """
    assert isinstance(month, int)
    assert month > 0
    assert month < 13

    # get the flux files of interest
    flux_files = generate_flux_filenames(
        directory_path=directory_path,
        file_prefix=file_prefix
    )

    # read in the fluxes
    tracer_path = directory_path + 'tracerinfo.dat'
    diag_path = directory_path + 'diaginfo.dat'
    fluxes = xbpch.open_mfbpchdataset(flux_files,
                                      dask=True,
                                      tracerinfo_file=tracer_path,
                                      diaginfo_file=diag_path)

    # find the time indices of interest
    if month + 1 < 10:
        time_idxs = np.where(
            fluxes.time.values < np.datetime64('1985-0%i-01' % (month + 1))
        )[0]
    else:
        time_idxs = np.where(
            fluxes.time.values < np.datetime64('1985-%i-01' (month + 1))
        )[0]

    # filter the fluxes and find the mean
    month_fluxes = fluxes[flux_field].values[time_idxs, :, :].mean(axis=0)

    return {
        'flux': month_fluxes,
        'latitude': fluxes.lat.values,
        'longitude': fluxes.lon.values,
        'time': fluxes.time.values[time_idxs]
    }
