"""
This script will generate N txt files with random gaussian noise, which can be
used as resampled prior means for the carbon flux inversion.

Author        : Mike Stanley
Created on    : September 22, 2020
Last Modified : September 22, 2020

================================================================================
OPERATIONAL PARAMETERS

Note, these variables are named to be consistent with the fortran code.

IIPAR (int)      : dimension of the longitude grid       | default 72
JJPAR (int)      : dimension of the latitude grid        | default 46
MMSCL (int)      : number of months in the optimization  | default 9
NNEMS (int)      : number of emissions                   | default 10
N     (int)      : number of txt files to generate       | default None
output_dir (str) : write location of files               | default None

"""
import argparse
import numpy as np
from scipy import stats
from tqdm import tqdm


def generate_prior_mean(
    iipar,
    jjpar,
    mmscl,
    nnems,
    mean=1,
    std=1.5,
    random_seed=None
):
    """
    Creates one draw from a Gaussian distribution.

    Parameters:
        iipar       (int)   : long dimension
        jjpar       (int)   : lat dimension
        mmscl       (int)   : number of optimization months
        nnems       (int)   : number of emissions
        mean        (float) : mean of distribution
        std         (float) : standard deviation of distribution
        random_seed (int)   : to make results reproducible

    Returns:
        prior_mean_sample (np arr) : (iipar x jjpar x mmscl x nnems,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    NUM_DRAWS = iipar * jjpar * mmscl * nnems
    prior_mean_sample = stats.norm(
        loc=mean,
        scale=std
    ).rvs(NUM_DRAWS)

    return prior_mean_sample


def generate_N_samples(
    N,
    output_dir,
    iipar,
    jjpar,
    mmscl,
    nnems,
    mean=1,
    std=1.5,
    random_seeds=None
):
    """
    Executes generate_prior_mean N times and writes to directory of choice.

    Files are labeled: prior_samp_#.txt

    Parameters:
        N            (int)   : number of samples to draw
        output_dir   (str)   : dump zone for generated data
        iipar        (int)   : long dimension
        jjpar        (int)   : lat dimension
        mmscl        (int)   : number of optimization months
        nnems        (int)   : number of emissions
        mean         (float) : mean of distribution
        std          (float) : standard deviation of distribution
        random_seeds (list)  : sequence of random states, default None

    Returns:
        None -- writes files to output_dir
    """
    if random_seeds is not None:
        assert len(random_seeds) == N

    for i in tqdm(range(N)):

        # generate the sample
        sample_i = generate_prior_mean(
            iipar=iipar,
            jjpar=jjpar,
            mmscl=mmscl,
            nnems=nnems,
            mean=mean,
            std=std,
            random_seed=random_seeds[i]
        )

        # write the file
        np.savetxt(
            fname=output_dir + '/prior_samp_%i.txt' % i,
            X=sample_i
        )


if __name__ == "__main__":

    # size defaults
    LON_DEF = 72
    LAT_DEF = 46
    MONTHS_DEF = 9
    NUM_EMS_DEF = 10

    # sample distribution defaults
    MEAN = 1
    STD = 1.5

    # random state default
    RANDOM_STATE = 42

    # initialize the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--IIPAR',
        help='longitude dimension',
        default=LON_DEF
    )
    parser.add_argument(
        '--JJPAR',
        help='latitude dimension',
        default=LAT_DEF
    )
    parser.add_argument(
        '--MMSCL',
        help='months dimension',
        default=MONTHS_DEF
    )
    parser.add_argument(
        '--NNEMS',
        help='emissions dimension',
        default=NUM_EMS_DEF
    )
    parser.add_argument(
        '--N',
        help='number of output files',
        default=None
    )
    parser.add_argument(
        '--output_dir',
        help='write location for output files',
        default=None
    )
    parser.add_argument(
        '--mean',
        help='sample distribution mean',
        default=MEAN
    )
    parser.add_argument(
        '--std',
        help='sample distribution std',
        default=STD
    )
    parser.add_argument(
        '--random_state',
        help='random state',
        default=RANDOM_STATE
    )

    args = parser.parse_args()

    # find random seeds
    if args.random_state is not None:
        random_seeds = np.random.choice(
            np.arange(0, 1000),
            size=args.N,
            replace=False
        )
    else:
        random_seeds = None

    # create the files
    generate_N_samples(
        N=args.N,
        output_dir=args.output_dir,
        iipar=args.IIPAR,
        jjpar=args.JJPAR,
        mmscl=args.MMSCL,
        nnems=args.NNEMS,
        mean=args.MEAN,
        std=args.STD,
        random_seeds=random_seeds
    )
