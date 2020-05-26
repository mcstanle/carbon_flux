"""
Given a directory of GOSAT observation files, this script counts the number of
observations in each grid box per month.

Saves a Mx72x46 numpy array, where M := is the number of months.

Author   : Mike Stanley
Created  : May 22, 2020
Modified : May 22, 2020

================================================================================
"""

from os.path import expanduser
import numpy as np
from tqdm import tqdm

from carbonfluxtools import io as cio
from carbonfluxtools import computation as ccomp

if __name__ == "__main__":

    # script configuration paraters
    BASE_DIR = expanduser('~') + '/Research/Carbon_Flux'
    SAT_DIR = BASE_DIR + '/data/GOSAT_OBS_2010'
    YEAR = 2010  # this is redundant since I made a separate directory
    SAVE_PATH = BASE_DIR + '/data/gosat_meta_data'

    # create a dataframe of the above observations
    gosat_df = cio.create_gosat_df_year(obs_dir=SAT_DIR, year=2010)

    # find the longitude/latitude indices for each observation
    lon_lat_idx = gosat_df.apply(
        lambda x: ccomp.lon_lat_to_IJ(lon=x['lon'], lat=x['lat']),
        axis=1
    )

    # instantiate a numpy array to hold grid counts for each month
    NUM_MONTHS = gosat_df['omonth'].nunique()
    grid_counts = np.zeros((NUM_MONTHS, 72, 46))

    for month_idx in tqdm(range(NUM_MONTHS)):

        # get the month mask
        month_mask = gosat_df['omonth'] == month_idx

        for idx, coord in lon_lat_idx.loc[month_mask].iteritems():

            # extract lon and lat
            lon_coord = coord[0]
            lat_coord = coord[1]

            # increment the above coordinate in grid counts
            grid_counts[month_idx, lon_coord, lat_coord] += 1

    # save the above array
    np.save(SAVE_PATH + '/2010_monthly_grid_counts.npy', grid_counts)
