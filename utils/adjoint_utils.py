"""
Utility code to deal with output and data from the geos-chem adjoint model.

Author:        Mike Stanley
Created:       November 18, 2019
Last Modified: November 18, 2019

================================================================================
"""

# import argparse
from datetime import datetime, timedelta
import glob
import pickle
# import sys

import numpy as np
import PseudoNetCDF as pnc

# define the user actions that can be taken
USER_ACTIONS = ['create_sf_dict']


def transform_time(inp_int_val, year=1985, month=1, day=1, hour=0):
    """
    Transforms a raw integer (as given in the bpch files) that represents hours
    since some time to an actual data time object.

    Parameters:
        inp_int_val (int) : integer represention of the time we wish to change
        year        (int) : index start year
        month       (int) : index start month
        day         (int) : index start day
        hour        (int) : index start hour

    Returns:
        transformed datetime value
    """
    assert isinstance(inp_int_val, int)

    start = datetime(year, month, day, hour)
    delta = timedelta(hours=inp_int_val)

    return start + delta


def create_sf_dict(dir_path, variable, output_path=None, year=2010):
    """
    Creates a dictionary of data from a collection of gctm.sf.** files. The
    output file is saved as a pickle.

    Parameters:
        dir_path    (str) : directory where code can find the scaling
        variable    (str) : CO2 scaling variable to use in the output
        output_path (str) : save location of output pickle file (if not none)
        year        (int) : starting year for inversion

    Returns:
        dictionary with the following key values
            - time      : numpy array
            - latitude  : numpy array
            - longitude : numpy array
            - sf_array  : numpy array

    NOTE:
    - PseudoNetcdf assumes that the tracerinfo.dat and diaginfo.dat files are
      included in the directory path given.

    TODO:
    - the time dimension in the sf files appear to all point to the same date.
    """
    # create the list of files
    sf_filepaths = sorted(glob.glob(dir_path + 'gctm.sf*'))

    # read in the above
    sf_files = [pnc.pncopen(path) for path in sf_filepaths]

    # get latitude/longitude/time information
    sample_file = sf_files[0]

    latitude = sample_file.variables['latitude'].array()
    longitude = sample_file.variables['longitude'].array()

    time_vals_raw = sample_file.variables['layer9'].array()
    time = [datetime(year, month, day=1) for month in time_vals_raw]

    # concatenate the scaling factors over time
    sf_concat = np.concatenate(
            [i.variables[variable].array() for i in sf_files]
    )

    output_dict = {
        'time': time,
        'latitude': latitude,
        'longitude': longitude,
        'sf_array': sf_concat
    }

    if output_path:
        with open(output_path, 'w') as f:
            pickle.dump(output_dict, f)

    return output_dict


# if __name__ == "__main__":

#     # initialize arg parser
#     parser = argparse.ArgumentParser()

#     # collect the first command line argument
#     user_action = sys.argv[1]

#     if user_action not in USER_ACTIONS:
#         raise ValueError('Please enter one of the following actions: %s' %
#                          str(USER_ACTIONS))

#     if user_action == 'create_sf_dict':

#         # add usage arguments
#         parser.add_argument('--dir_path',
#                             help='directory path where gctm.sf** files are')
#         parser.add_argument('--out_path',
#                             help='output location for pickle')

#         # parse the arguments
#         args, unknown = parser.parse_known_args()

#         # create the dictionary and save as pickle
#         create_sf_dict(dir_path=args.dir_path, output_path=args.out_path)
