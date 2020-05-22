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

from carbonfluxtools import io as cio

if __name__ == "__main__":

    # script configuration paraters
    BASE_DIR = expanduser('~') + '/Research/Carbon_Flux'
    SAT_DIR = BASE_DIR + '/data/GOSAT_OBS_2010'
    YEAR = 2010  # this is redundant since I made a separate directory

    # create a dataframe of the above observations
    gosat_df = cio.create_gosat_df_year(obs_dir=SAT_DIR, year=2010)

    print(gosat_df.head())
