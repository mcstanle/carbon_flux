"""
This script should be able to do the following
1. generate ground truth gosat observations
2. generate set of perturbations for gosat observations

Author        : Mike Stanley
Created       : April 2, 2020
Last Modified : April 15, 2020

===============================================================================
- USE
I want to be able to pass
1. base path where GOSAT observations are located
2. data range of interest
    - format YYYYMMDD
3. modeled XCO2 file

and then write to another given directory.

Output files should have the form:
- GOSAT_OSSE_YYYYMMDD.txt
This is to match the expected input in the geos-chem code. Different forms of
the input code should be organized by directory structure.

===============================================================================
"""

import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pathlib
import PseudoNetCDF as pnc
from tqdm import tqdm

# custom functions
from read_gosat_obs import read_gosat_data


def read_modeled_co2(file_path, variable_oi='IJ-AVG-$_CO2'):
    """
    Reads bpch file from GCAdj output to get the modeled XCO2 values.

    Parameters:
        file_path   (str)
        variable_oi (str) : name of variable in bpch file for modeled XCO2

    Returns:
        tuple - (numpy arr of modeled values, lon, lat)

    NOTE:
    - assume first dimension of model_file is ignorable
    """
    # read in the model file
    model_file = pnc.pncopen(file_path, format='bpch')

    # find lat/lon arrays
    lat = model_file.variables['latitude'].array()
    lon = model_file.variables['longitude'].array()

    return model_file.variables[variable_oi].array()[0, :, :, :], lon, lat


def lon_lat_to_IJ(lon, lat, lon_size=5, lat_size=4):
    """
    Transform (lon, lat) coordinates to 4x5 grid

    Parameters:
        lon      (float) : longitude
        lat      (float) : latitude
        lon_size (int)   : number of degrees in lon grid
                           default is 5
        lat_size (int)   : number of degress in lat grid
                           default is 4

    Returns:
        (I, J) longitude/latitude coordinates

    NOTES:
    - this code is copied from the ./code/modified/grid_mod.f
      - the primary difference is that python arrays are indexed from 0
    """
    LON_idx = int((lon + 180) / lon_size + .5)
    LAT_idx = int((lat + 90) / lat_size + .5)

    if LON_idx >= 72:
        LON_idx = LON_idx - 72

    return LON_idx, LAT_idx


def alter_xco2(gosat_obs, new_xco2):
    """
    Expects a GOSAT file consistent with the 7-part format

    Parameters:
        gosat_obs (list)

    Returns
        list with same format as gosat_obs but with the new xco2 value

    NOTE:
    - make sure to send in copy of gosat_obs
    """
    assert len(gosat_obs) == 7

    # create new list for altered observation
    altered_obs = [sub_obs[:] for sub_obs in gosat_obs]

    # alter the XCO2 value
    altered_obs[0][4] = new_xco2

    return altered_obs


def create_gosat_day(obs_list, modeled_obs):
    """
    Creates a new GOSAT observation day file. These are organized
    such that one observation occurs every 7 elements in the list.

    Parameters:
        obs_list    (list)      : all observations
        modeled_obs (numpy arr) : lon X lat grid of modeled observations

    Returns:
        New list of the form obs_list -- new meaning new XCO2 values

    NOTES:
    - assumes 72 longitude elements and 46 latitude elements
    - assumes that the modeled_obs arrives as lat X lon
    """
    assert modeled_obs.shape[0] == 46  # check latitude dimesion
    assert modeled_obs.shape[1] == 72    # check longitude dimesion

    # create indices to loop over
    start_idx = np.arange(0, len(obs_list), step=7)
    end_idx = np.arange(7, len(obs_list) + 1, step=7)

    assert len(start_idx) == len(end_idx)

    obs_idx = zip(start_idx, end_idx)

    altered_obs = []
    for start, end in obs_idx:

        # pull out the GOSAT observation
        gosat_obs = obs_list[start:end]

        # find array indices of real observation
        lon_idx, lat_idx = lon_lat_to_IJ(
            lon=gosat_obs[0][1],
            lat=gosat_obs[0][2],
            lon_size=5,
            lat_size=4
        )

        altered_obs.extend(
            alter_xco2(
                gosat_obs=gosat_obs,
                new_xco2=modeled_obs[lat_idx, lon_idx]
            )
        )

    assert len(altered_obs) == len(obs_list)
    return altered_obs


def create_perturbed_gosat_day(obs_list, perturb_arr):
    """
    Given a day of GOSAT observations and a perturbation array, create a new
    GOSAT day.

    Parameters:
        obs_list    (list)      : all observations from day
        perturb_arr (numpy arr) : 1 perturbation for each obs during day

    Returns
        New list of the form obs_list -- new meaning new XCO2 values
    """
    assert len(obs_list) / 7 == len(perturb_arr)

    # create indices to loop over
    start_idx = np.arange(0, len(obs_list), step=7)
    end_idx = np.arange(7, len(obs_list) + 1, step=7)

    assert len(start_idx) == len(end_idx)

    obs_idx = zip(start_idx, end_idx)

    perturbed_obs = []
    idx = 0
    for start, end in obs_idx:

        # pull out the GOSAT observation
        gosat_obs = obs_list[start:end]

        # perturb the observation
        perturbed_val = gosat_obs[0][4] + perturb_arr[idx]

        perturbed_obs.extend(
            alter_xco2(
                gosat_obs=gosat_obs,
                new_xco2=perturbed_val
            )
        )

        idx += 1

    assert len(perturbed_obs) == len(obs_list)
    return perturbed_obs


def write_gosat_day(obs_list, write_path):
    """
    Writes a gosat observation list to file

    Parameters:
        obs_list   (list) : list of observations as created by
                            create_gosat_day()
        write_path (str)  : file path to file
        precision  (int)  : number of decimal points that the value is carried
                            until

    Returns:
        write_path

    NOTES:
    - for each row written, the files should be comma delimited
    - we hard-code 5 decimal point precision
    """
    path = pathlib.Path(write_path)

    assert path.parents[0].is_dir()   # make sure the given directory exists

    with open(path, mode='w') as f:
        for line in obs_list:
            f.write(
                ', '.join([f"{i:.5f}" for i in line]) + '\n'
            )

    return write_path


def generate_unpert_gosat_files(start_date, end_date, origin_dir, save_dir,
                                modeled_xoc2, lon, lat,
                                gosat_file_form='GOSAT_OSSE_'):
    """
    Creates new unperturbed GOSAT files.

    Parameters:
        start_date      (str)       : YYYYMMDD
        end_date        (str)       : YYYYMMDD (non - inclusive)
        origin_dir      (str)       : directory where original files are
                                      located
        save_dir        (str)       : directory for saving files
        modeled_xoc2    (numpy arr) : new XCO2 values
        lon             (numpy arr) :
        lat             (numpy arr) :
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

    # verify that the number of days matches the number of daya in modeled_xco2
    assert (end_dt - start_dt).days == modeled_xoc2.shape[0]

    # get all dates of interest
    dates = [datetime.strftime(i, '%Y%m%d')
             for i in pd.date_range(start=start_dt, end=end_dt)]

    # create input/output paths
    input_files = [
        origin_dir + '/' + gosat_file_form + date + '.txt'
        for date in dates
    ][:-1]
    output_files = [
        save_dir + '/' + gosat_file_form + date + '.txt'
        for date in dates
    ][:-1]

    # for each file generate new file - one per day!
    for idx, gosat_file_nm in tqdm(enumerate(input_files)):

        # check to see if generated date has origin file
        if not os.path.exists(gosat_file_nm):
            continue

        # read in the original observation
        gos_orig = read_gosat_data(fp=gosat_file_nm)

        # create a new observation list with the modeled XCO2
        gos_new = create_gosat_day(
            obs_list=gos_orig,
            modeled_obs=modeled_xoc2[idx, :, :] / 1e3      # TODO find out why
        )

        # write the new file
        write_gosat_day(
            obs_list=gos_new,
            write_path=output_files[idx]
        )
        # print('Written: %s' % output_files[idx])


def generate_pert_gosat_files(start_date, end_date, origin_dir, save_dir,
                              perturb_dir, perturb_idx,
                              gosat_file_form='GOSAT_OSSE_'):
    """
    Creates new perturbed GOSAT files. Since each perturbation file has several
    perturbation options, which exist on the rows of the perturbation array,
    perturb_idx gives a 0-indexed indication as to which should be used.

    Parameters:
        start_date      (str)       : YYYYMMDD
        end_date        (str)       : YYYYMMDD (non - inclusive)
        origin_dir      (str)       : directory where original files are
                                      located
        save_dir        (str)       : directory for save perturbed GOSAT files
        perturbed_dir   (str)       : directory where perturbation files are
        perturb_idx     (int)       : identifies the perturbation to use
        gosat_file_form (str)       :

    Returns:
        Saves new files in origin_dir with format

    NOTE:
    - we assume the GOSAT file format
    """
    assert os.path.isdir(origin_dir)
    assert os.path.isdir(save_dir)
    assert os.path.isdir(perturb_dir)
    # assert perturb_idx is not None

    # change strings into datetime objects
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    assert start_dt < end_dt

    # get all dates of interest
    dates = [datetime.strftime(i, '%Y%m%d')
             for i in pd.date_range(start=start_dt, end=end_dt)]

    # create input/output paths
    input_files = [
        origin_dir + '/' + gosat_file_form + date + '.txt'
        for date in dates
    ][:-1]
    output_files = [
        save_dir + '/' + gosat_file_form + date + '.txt'
        for date in dates
    ][:-1]
    perturbed_files = [
        perturb_dir + '/' + date + '_perturb.npy'
        for date in dates
    ][:-1]

    # for each file generate new file - one per day!
    for idx, gosat_file_nm in tqdm(enumerate(input_files)):

        # check to see if generated date has origin file
        if not os.path.exists(gosat_file_nm):
            continue

        # read in the original observation
        gos_orig = read_gosat_data(fp=gosat_file_nm)

        # read in perturb arr
        perturb_arr = np.load(file=perturbed_files[idx])
        assert perturb_idx < perturb_arr.shape[0]
        perturb_arr = perturb_arr[perturb_idx, :]

        # create a new observation list with the modeled XCO2
        gos_pert = create_perturbed_gosat_day(
            obs_list=gos_orig,
            perturb_arr=perturb_arr
        )

        # write the new file
        write_gosat_day(
            obs_list=gos_pert,
            write_path=output_files[idx]
        )
        # print('Written: %s' % output_files[idx])


if __name__ == '__main__':

    # default values
    BASE_PATH = '/Users/mikestanley/Research/Carbon_Flux'
    BASE_FILE_D = BASE_PATH + '/gc_adj_tutorial/OSSE_OBS'
    DATE_LB = '20100101'
    DATE_UB = '20100901'
    MODELED_XCO2 = BASE_PATH + '/data/forward_model_output/satellite_obs/\
gosat_JULES_201001_201009/gctm.model.01'
    OUTPUT_DIR = BASE_PATH + '/data/modeled_satellite_obs/JULES_unpert'
    PERTURB = True
    PERTURB_DIR = BASE_PATH + '/data/modeled_satellite_obs/pert_files'

    # initialize the argparser
    parser = argparse.ArgumentParser()

    # fundamental arguments
    parser.add_argument('--base_file_dir',
                        default=BASE_FILE_D, type=str)
    parser.add_argument('--date_lb',
                        default=DATE_LB, type=str)
    parser.add_argument('--date_ub',
                        default=DATE_UB, type=str)
    parser.add_argument('--modeled_XCO2_path',
                        default=MODELED_XCO2, type=str)
    parser.add_argument('--output_dir',
                        default=OUTPUT_DIR, type=str)

    # perturbation arguments
    parser.add_argument('--perturb', type=bool, default=PERTURB)
    parser.add_argument('--perturb_dir', type=str, default=PERTURB_DIR)
    parser.add_argument('--perturb_idx', type=int, default=None)

    # parse the given arguments
    args = parser.parse_args()

    if args.perturb:

        # create perturbed files
        generate_pert_gosat_files(
            start_date=args.date_lb,
            end_date=args.date_ub,
            origin_dir=args.base_file_dir,
            save_dir=args.output_dir,
            perturb_dir=args.perturb_dir,
            perturb_idx=args.perturb_idx
        )

        # log the first XCO2 value
        file_names = os.listdir(args.output_dir)
        first_gosat_file = read_gosat_data(
            fp=args.output_dir + '/' + file_names[1]
        )

        with open(args.output_dir + '/log.txt', 'a') as f:
            f.write(str(datetime.now()) + '\t' +
                    str(first_gosat_file[0][4]))
            f.write('\n')

    else:

        # read in modeled XCO2 file
        model_xco2_arr, lon, lat = read_modeled_co2(
            file_path=args.modeled_XCO2_path
        )
        print('Modeled values read in')

        # generate new GOSAT 0bservation files
        generate_unpert_gosat_files(
            start_date=args.date_lb,
            end_date=args.date_ub,
            origin_dir=args.base_file_dir,
            save_dir=args.output_dir,
            modeled_xoc2=model_xco2_arr,
            lon=lon,
            lat=lat
        )
