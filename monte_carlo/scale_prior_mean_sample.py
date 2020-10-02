"""
This script can access a directory of prior mean samples and scale them by some
value. The value of doing this is that we can adjust the variance of the sample
which applies to the scenario in which we want to perform a monte carlo run
with different variance, but using the same strucure.

Author        : Mike Stanley
Created       : Oct 2, 2020
Last Modified : Oct 2, 2020
"""
import argparse
from glob import glob
import numpy as np
from tqdm import tqdm


def translate_file(input_path, output_path, scale_factor):
    """
    Read in a prior mean text file. Expects that each line of the file
    corresponds to only one sample.

    Parameters:
        input_path   (str)   : path to the file of interest
        output_path  (str)   : where new scaled file should be written
        scale_factor (float) : factor by which scale each sample in file

    Returns:
        None : writes a file to the output path
    """
    # read in the file
    prior_mean_arr = np.loadtxt(fname=input_path)

    # multiply the entire array by scale factor
    prior_mean_arr = scale_factor * prior_mean_arr

    # write the new array to file
    np.savetxt(fname=output_path, X=prior_mean_arr)


if __name__ == "__main__":

    # scaling value
    SCALE = 2

    # direcoties info
    BASE_DIR = '/glade/work/mcstanley/monte_carlo/V1/data'
    SOURCE_DIR = BASE_DIR + '/prior_means'
    DEST_DIR = BASE_DIR + '/prior_means_scl3'

    # obtain the sample file names
    sample_file_nms = glob(SOURCE_DIR + '/*')

    print(sample_file_nms)

    for file_path in tqdm(sample_file_nms):

        # create an output path
        output_path = DEST_DIR + '/' + file_path.split('/')[-1]

        # scale the prior mean
        translate_file(
            input_path=file_path,
            output_path=output_path,
            scale_factor=SCALE
        )
