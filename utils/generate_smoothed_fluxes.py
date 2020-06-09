"""
This script is primarily responsible for removing the diurnal cycle from
fluxes of interest.

INPUT :
    - change FLUX_DIR

OUTPUT :
    - change OUTPUT_DIR

Output files are txt files. When unpacking, indices move from fastest to
slowest in the following order
1. latitude
2. longitude
3. 3hr steps

Author        : Mike Stanley
Created       : June 2, 2020
Last Modified : June 8, 2020
"""

from glob import glob
import numpy as np
import pandas as pd
import xbpch

import carbonfluxtools.io as cio


def run(
    flux_dir,
    flux_prefix,
    tracerinfo_path,
    diaginfo_path,
    output_dir,
    window_size=8,
    varname_oi='CO2_SRCE_CO2bf'
):
    """
    Perform the steps requires to create the new flux files.

    Parameters:
        flux_dir        (str) : directory where daily flux files are found
        flux_prefix     (str) : prefix of flux files
        tracerinfo_path (str) : path to tracerinfo file for bpch read
        diaginfo_path   (str) : path to diaginfo file for bpch read
        output_dir      (str) : output directory location for new fluxes
        window_size     (int) : window size for smoothing
        varname_oi      (str) : variable of interest in the bpch files

    Output:
        creates smoothed daily flux files
    """
    # obtain the sorted flux file names
    flux_files = sorted(
        glob(flux_dir + '/' + flux_prefix + '*'),
        key=lambda x: int(x[-3:])
    )

    # read in the fluxes -- BPCH FILES
    # flux_data = xbpch.open_mfbpchdataset(
    #     flux_files,
    #     dask=True,
    #     tracerinfo_file=tracerinfo_path,
    #     diaginfo_file=diaginfo_path
    # )

    # extract flux arrays from the xbpch objects
    # flux_arr = flux_data.variables[varname_oi].values

    # read in the fluxes -- TXT FILES
    flux_arr = cio.read_flux_txt_files(flux_files=flux_files)

    print('Fluxes acquired')

    # get the dimensions of the above array for reshaping
    arr_dims = flux_arr.shape

    # reshape the above array to prepare for dataframe
    flux_2d = flux_arr.flatten().reshape(
        arr_dims[0], arr_dims[1] * arr_dims[2]
    )

    # make the above into a dataframe
    flux_df = pd.DataFrame(flux_2d)

    # find the rolling average over time
    flux_df_ra = flux_df.rolling(
        window=window_size, axis=0
    ).mean().interpolate(limit_direction='both')

    # reshape back to original size
    flux_arr_new = flux_df_ra.values.flatten().reshape(
        arr_dims[0], arr_dims[1], arr_dims[2]
    )

    assert flux_arr_new.shape == arr_dims

    # write the new flux array to directory
    cio.generate_txt_files_np(
        flux_arr=flux_arr_new,
        bpch_files=flux_files,
        output_dir=output_dir
    )
    print('Done.')


if __name__ == "__main__":

    # define I/O constants
    BASE_DIR = '/Users/mikestanley/Research/Carbon_Flux'
    FLUX_DIR = BASE_DIR + '/data/JULES_YEAR_txt'

    # define prefixes
    FLUX_PREFIX = 'nep.geos.4x5.2010.'

    # tracer and diag paths
    TRACERINFO_PATH = FLUX_DIR + '/tracerinfo.dat'
    DIAGINFO_PATH = FLUX_DIR + '/diaginfo.dat'

    # output directory
    OUTPUT_DIR = BASE_DIR + '/data/JULES_smooth'

    # create the new smoothed fluxes
    run(
        flux_dir=FLUX_DIR,
        flux_prefix=FLUX_PREFIX,
        tracerinfo_path=TRACERINFO_PATH,
        diaginfo_path=DIAGINFO_PATH,
        output_dir=OUTPUT_DIR,
    )
