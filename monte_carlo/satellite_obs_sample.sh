#!/bin/bash
#
# Takes unperturbed GOSAT observations and perturbation files and creates new
# perturned satellite observations.
#
# Author        : Mike Stanley
# Created       : September 22, 2020
# Last Modified : September 22, 2020
# =============================================================================

# operations parameters
NUM_ELEMENTS=5
CODE_LOC=/glade/u/home/mcstanley/carbon_flux/utils
BASE_GOSAT_DIR=/glade/work/mcstanley/Data/modeled_satellite_obs
UNPERT_GOSAT_SOURCE=$BASE_GOSAT_DIR/JULES_unpert
PERTURB_FILE_SOURCE=$BASE_GOSAT_DIR/pert_files
BASE_OUTPUT_DIR=/glade/work/mcstanley/monte_carlo/V1/data/satellite_obs

DATE_LB="20100101"
DATE_UB="20100901"

# generate the observations
for i in $(seq 0 $( expr ${NUM_ELEMENTS} - 1))
do

    # define the satellite directory
    SAT_DATA_DIR=$BASE_OUTPUT_DIR/samp_${i}
    mkdir -p $SAT_DATA_DIR

    python $CODE_LOC/generate_gosat_data.py        \
    --base_file_dir=$UNPERT_GOSAT_SOURCE           \
    --date_lb=$DATE_LB                             \
    --date_ub=$DATE_UB                             \
    --output_dir=$SAT_DATA_DIR                     \
    --perturb_dir=$PERTURB_FILE_SOURCE             \
    --perturb_idx=${i}
   
done