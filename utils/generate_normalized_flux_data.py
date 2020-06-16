"""
Accepts two sets of fluxes and scales one of them so that the two fluxes have
the same total flux over their time domain.

Author        : Mike Stanley
Created       : June 2, 2020
Last Modified : June 2, 2020
"""

from glob import glob
import numpy as np
import xbpch

import carbonfluxtools.io as cio
import carbonfluxtools.computation as ccomp


def run(
    bpch_use,
    true_flux_dir,
    prior_flux_dir,
    true_flux_prefix,
    prior_flux_prefix,
    lat_lon_dir,
    tracerinfo_path,
    diaginfo_path,
    output_dir,
    varname_oi='CO2_SRCE_CO2bf',
    TOL=0.000001
):
    """
    Run the steps required to make the scaled fluxes and write them to disk.

    At the moment, this function scales the prior down to have the same global
    flux as the truth.

    Parameters:
        bpch_use          (bool)  : switch to indicate that bpch files are
                                    flux input
        true_flux_dir     (str)   : directory location of true fluxes
        prior_flux_dir    (str)   : directory location of prior fluxes
        true_flux_prefix  (str)   : e.g. 'nep.geos.4x5.2010.'
        prior_flux_prefix (str)   : e.g. 'nep.geos.4x5.'
        lat_lon_dir       (str)   : directory where lat/lon arrays can be found
                                    use when bpch_use==True
        tracerinfo_path   (str)   : location of tracerinfo file for
                                    reading bpch
        diaginfo_path     (str)   : location of diaginfo file for reading bpch
        output_dir        (str)   : directory of txt output files
        varname_oi        (str)   : variable to extract from the bpch objects
        TOL               (float) : tolerance of new integrated flux

    Returns:
        writes daily flux txt files to output_dir
    """
    # find the sorted file names
    prior_files = sorted(
        glob(prior_flux_dir + '/' + prior_flux_prefix + '*'),
        key=lambda x: int(x[-3:])
    )
    true_files = sorted(
        glob(true_flux_dir + '/' + true_flux_prefix + '*'),
        key=lambda x: int(x[-3:])
    )

    if bpch_use:

        # read in the fluxes
        prior_data = xbpch.open_mfbpchdataset(
            prior_files,
            dask=True,
            tracerinfo_file=tracerinfo_path,
            diaginfo_file=diaginfo_path
        )
        print('Prior fluxes acquired')
        true_data = xbpch.open_mfbpchdataset(
            true_files,
            dask=True,
            tracerinfo_file=tracerinfo_path,
            diaginfo_file=diaginfo_path
        )
        print('True fluxes acquired')

        # extract flux arrays from the xbpch objects
        prior_arr = prior_data.variables[varname_oi].values
        true_arr = true_data.variables[varname_oi].values

        # get longitude/latitude arrays
        lons = prior_data.variables['lon'].values
        lats = prior_data.variables['lat'].values

    else:

        # read in the fluxes
        prior_arr = cio.read_flux_txt_files(flux_files=prior_files)
        true_arr = cio.read_flux_txt_files(flux_files=true_files)

        # read in lat/lon
        lons = np.load(lat_lon_dir + '/lon.npy')
        lats = np.load(lat_lon_dir + '/lat.npy')

    print('=== Flux array dimensions ===')
    print('Prior : %s' % str(prior_arr.shape))
    print('Truth : %s' % str(true_arr.shape))
    print('Lon   : %s' % str(lons.shape))
    print('Lat   : %s' % str(lats.shape))

    # determine global integral of prior and posterior
    prior_global_flux = ccomp.compute_global_flux(
        flux_arr=prior_arr, lons=lons, lats=lats
    )
    true_global_flux = ccomp.compute_global_flux(
        flux_arr=true_arr, lons=lons, lats=lats
    )

    # find the scalar multiplier
    scl_mult = true_global_flux / prior_global_flux
    print('Scalar multiplier : %.5f' % scl_mult)

    # scale the prior
    prior_arr_scl = scl_mult * prior_arr

    # compute new integrated flux
    prior_global_flux_scl = ccomp.compute_global_flux(
        flux_arr=prior_arr_scl, lons=lons, lats=lats
    )
    scl_mult_updated = true_global_flux / prior_global_flux_scl

    if np.abs(scl_mult_updated - 1) > TOL:
        print('Normalized flux exceeds tolerance: TOL = %.10f' % np.abs(
            scl_mult_updated - 1)
        )
        exit()

    # write the new flux array to directory
    cio.generate_txt_files_np(
        flux_arr=prior_arr_scl,
        bpch_files=prior_files,
        output_dir=output_dir
    )


if __name__ == "__main__":

    # define some I/O constants
    BASE_DIR = '/Users/mikestanley/Research/Carbon_Flux'
    PRIOR_FLUX_DIR = BASE_DIR + '/data/NEE_fluxes'
    TRUE_FLUX_DIR = BASE_DIR + '/data/JULES'

    # using bpch files?
    BPCH_USE = False
    LAT_LON_DIR = BASE_DIR + '/data/lon_lat_arrs'

    # define prefixes
    PRIOR_FLUX_PREFIX = 'nep.geos.4x5.'
    TRUE_FLUX_PREFIX = 'nep.geos.4x5.2010.'

    # tracer and diag paths
    TRACERINFO_PATH = PRIOR_FLUX_DIR + '/tracerinfo.dat'
    DIAGINFO_PATH = PRIOR_FLUX_DIR + '/diaginfo.dat'

    # output directory
    OUTPUT_DIR = BASE_DIR + '/data/NEE_fluxes_txt_scl'

    # create new scaled fluxes
    run(
        bpch_use=BPCH_USE,
        true_flux_dir=TRUE_FLUX_DIR,
        prior_flux_dir=PRIOR_FLUX_DIR,
        true_flux_prefix=TRUE_FLUX_PREFIX,
        prior_flux_prefix=PRIOR_FLUX_PREFIX,
        lat_lon_dir=LAT_LON_DIR,
        tracerinfo_path=TRACERINFO_PATH,
        diaginfo_path=DIAGINFO_PATH,
        output_dir=OUTPUT_DIR
    )
