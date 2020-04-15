"""
Given a directory of GOSAT observations, this script is meant to generate
perturbations for each observation in each file for some date interval.

Author        : Mike Stanley
Created       : April 3, 2020
Last Modified : April 3, 2020

===============================================================================
Assumptions:
1. the XCO2 uncertainty found in the GOSAT files is a variance term

"""

import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd
from scipy import stats
from tqdm import tqdm

# custom functions
from read_gosat_obs import read_gosat_data


def extract_uncert(gosat_list):
    """
    Extracts the XCO2_unc value from each observation in a GOSAT file. For each
    observation, there are 7 data pieces. The XCO2_unc component is found in
    the 6th component of the first data piece.

    Parameters:
        gosat_list (list) : list of all gosat data observables

    Returns:
        list of XCO2 uncertainies; one for each observation in the given file.
    """
    # find all of the first pieces of data for each observation
    core_obs_idxs = np.arange(0, len(gosat_list), step=7)

    # recover the errors for each observation
    obs_errs = np.array([gosat_list[idx][5] for idx in core_obs_idxs])

    return obs_errs


def generate_uncertainty_array(obs_errs, num_samples, rand_seed):
    """
    Generates num_samples rows in a 2D array where each column represents an
    observation and each row is a sampled value from the gaussian error
    distribution, whose variance is defined by the obs_errs array.

    Parameters:
        obs_errs    (numpy arr) : error variance for each observation
        num_samples (int)       : number of perturbations to draw from distr
        rand_seed   (int)       : random seed to make these results
                                  reproducible.

    Returns:
        numpy array with NxK elements
            - N number of perturbs
            - K number of observations
    """
    # perturbation generator
    num_obs = len(obs_errs)
    pert_gen = stats.norm(loc=np.zeros(num_obs), scale=np.sqrt(obs_errs))

    # generate perturbations
    np.random.seed(rand_seed)
    perts = pert_gen.rvs(size=(num_samples, num_obs))

    return perts


def generate_perturbation_files(start_date, end_date, origin_dir, save_dir,
                                num_perts,
                                gosat_file_form='GOSAT_OSSE_'):
    """
    Creates new perturbation files given time range and a seed directory.

    Parameters:
        start_date      (str)       : YYYYMMDD
        end_date        (str)       : YYYYMMDD (non - inclusive)
        origin_dir      (str)       : directory where original files are
                                      located
        save_dir        (str)       : directory for saving files
        num_perts       (int)       : number of perturbations to make for
                                      each observation.
        gosat_file_form (str)       :

    Returns:
        Saves new files in origin_dir with format

    NOTE:
    - the number of days (first dimension in modeled_xco2 should equal number
      days considered by the date range)
    - we assume the GOSAT file format
    """
    assert os.path.isdir(origin_dir)
    assert os.path.isdir(save_dir)

    # change strings into datetime objects
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    assert start_dt < end_dt

    # get all file names in origin directory
    dates = [datetime.strftime(i, '%Y%m%d')
             for i in pd.date_range(start=start_dt, end=end_dt)]

    # create input/output paths
    input_files = [
        origin_dir + '/' + gosat_file_form + date + '.txt'
        for date in dates
    ][:-1]
    output_files = [
        save_dir + '/' + date + '_perturb' + '.npy'
        for date in dates
    ][:-1]

    # for each seed file, generate a different random seed
    random_seeds = np.arange(0, len(input_files))

    # for each file generate new file - one per day!
    for idx, gosat_file_nm in tqdm(enumerate(input_files)):

        # continue if the generated observation file path does not exist
        if not os.path.exists(gosat_file_nm):
            continue

        # read in the original observation
        gos_obs = read_gosat_data(fp=gosat_file_nm)

        # extract uncertainty information
        obs_errs = extract_uncert(gosat_list=gos_obs)

        # generate perturbations
        unc_arr = generate_uncertainty_array(
            obs_errs=obs_errs,
            num_samples=num_perts,
            rand_seed=random_seeds[idx]
        )

        # write the new file
        np.save(
            file=output_files[idx],
            arr=unc_arr
        )


if __name__ == '__main__':

    # constants for file use
    BASE_PATH = '/Users/mikestanley/Research/Carbon_Flux'
    BASE_FILE_D = BASE_PATH + '/gc_adj_tutorial/OSSE_OBS'
    DATE_LB = '20100101'
    DATE_UB = '20100901'
    OUTPUT_DIR = BASE_PATH + '/data/modeled_satellite_obs/pert_files'
    NUM_PERTS = 1000

    # initialize the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gosat_file_dir',
                        default=BASE_FILE_D)
    parser.add_argument('--date_lb',
                        default=DATE_LB)
    parser.add_argument('--date_ub',
                        default=DATE_UB)
    parser.add_argument('--output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('--num_perts',
                        default=NUM_PERTS)

    # parse the given arguments
    args = parser.parse_args()

    # generate the files
    generate_perturbation_files(
        start_date=args.date_lb,
        end_date=args.date_ub,
        origin_dir=args.gosat_file_dir,
        save_dir=args.output_dir,
        num_perts=args.num_perts
    )
