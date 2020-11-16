#!/bin/bash
# This script kicks off a single element of the monte carlo procedure.
#
# This is where parameters of each individual inversion should be updated, like
# - number of iterations (NUM_ITER)
# - number of threads (NUM_THREADS)
# - run directory (RUN_DIR)
# - satellite data directory (SAT_DATA_DIR)
#
# Author        : Mike Stanley
# Created       : September 21, 2020
# Last Modified : November 16, 2020
#==============================================================================

# define operational variables
ENS_NUM=0
BASE_DIR=/glade/work/mcstanley
export NUM_ITER=11
export RUN_NAME=ens_${ENS_NUM}
export RUN_DIR=$BASE_DIR/monte_carlo/multijob
export NUM_THREADS=9

# data locations
export PRIOR_MEAN_FILE=$RUN_DIR/data/prior_means_scl3/prior_samp_${ENS_NUM}.txt
export SAT_DATA_DIR=$RUN_DIR/data/satellite_obs/samp_${ENS_NUM}
export START_VEC_LOC=$BASE_DIR/Data/baseline_start_vectors/pu3_baseline_start.txt

# run the inversion
ELEMENT_OUT_DIR=$RUN_DIR/$RUN_NAME/runs/v8-02-01/geos5

#mpiexec_mpt 
bash $ELEMENT_OUT_DIR/run 1> $ELEMENT_OUT_DIR/stdout.txt 2> $ELEMENT_OUT_DIR/stderr.txt
