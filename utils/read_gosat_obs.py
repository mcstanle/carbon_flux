"""
This script is designed to provide the utility functions necessary to read in
simulated GOSAT txt files in a jupyter notebook. It should be able to take in
directory paths, and a range of dates for files to read in.

Author:        Mike Stanley
Created:       Jan 27, 2020
Last Modified: Jan 27, 2020
"""

from glob import glob
import numpy as np
import pandas as pd

# constants
COLUMN_NMS = [
    'lgos',
    'lon',
    'lat',
    'psurf',
    'xco2',
    'xco2_unc',
    'oyear',
    'omonth',
    'oday',
    'ohour',
    'omin',
    'osec',
    'gcpsurf',
    'xco2_unce',
    'date_field1',
    'date_field2',
    'date_field3',
    'date_field4',
    'date_field5',
    'date_field6'
]


def read_gosat_data(fp):
    """
    Read in and process GOSAT Data.

    Parameters:
        fp (str) : file path to GOSAT Obs

    Returns:
        List of lists with satellite observations
    """
    # read in the file
    gs_data = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            gs_data.append(line.replace('\n', '').split(', '))

    # convert strings to floats
    gs_data = [[float(num) for num in line] for line in gs_data]

    return gs_data


def create_gosat_df(fp, column_list=COLUMN_NMS):
    """
    Read in GOSAT file using read_gosat_data and then creates a pandas
    dataframe with the following columns:
    1. lon
    2. lat
    3. xco2
    4. xco2 uncertainty
    5.-10. year, month, day, hour, min, sec

    Note, the key observation used to establish this data is that the meat of
    each observation is on every 7th line.

    Parameters:
        fp           (str) : path to file
        column_list (list) : list of column names, specified above

    Returns:
        pandas dataframe with the above columns
    """
    # read in the file
    raw_gosat_file = read_gosat_data(fp)

    # get the observation indices
    obs_idxs = np.arange(0, len(raw_gosat_file), 7)

    # create a shorted observation list
    obs_list = [raw_gosat_file[idx] for idx in obs_idxs]

    # return a pandas dataframe
    return pd.DataFrame(obs_list, columns=column_list)


def create_gosat_df_year(obs_dir, year, column_list=COLUMN_NMS):
    """
    reads in all simulated GOSAT observation files from directory for a given
    year and generates a pandas dataframe with the following columns:
    1. lon
    2. lat
    3. xco2
    4. xco2 uncertainty
    5.-10. year, month, day, hour, min, sec

    Parameters:
        obs_dir     (str) : directory with GOSAT observations of interest
        year        (int) : year of observation
        column_list (str) : list of column names, specified above

    Returns:
        pandas dataframe with the above columns
    """
    # get all file names in the given directory
    all_fps = glob(obs_dir + '*')

    # only keep those from our year of interest
    fps_year = [i for i in all_fps if int(i[-12:-8]) == year]

    # for each file path, apply create_gosat_df()
    obs_dfs = [create_gosat_df(fp, column_list=COLUMN_NMS) for fp in fps_year]

    # concatenate these dfs together
    obs_df = pd.concat(obs_dfs, ignore_index=True)

    # sort the above by time stamps
    obs_df.sort_values(['oyear', 'omonth', 'oday'], inplace=True)

    return obs_df.reset_index(drop=True)
