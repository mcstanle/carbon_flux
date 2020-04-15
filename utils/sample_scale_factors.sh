#!/bin/bash
# this script is meant to leverage the generate_gosat_data.py file in order to
# iteratively sample from GOSAT and generate the optimized scale factors from
# GEOS-Chem adjoint
#
# Author        : Mike Stanley
# Created       : April 3, 2020
# Last Modified : April 3, 2020
#
# ---- NOTES ---- 
# 1. the python file generate_gosat_data.py needs to be in the same directory as this script.
# 2. Make sure that the date bounds match the bounds in ./input.geos
# 3. every time that generate_gosat_data.py is run, it pulls the first XCO2 value and writes
#    it to a log file so that we can confirm the perturbed files are indeed changing.

# ========= OPERATIONAL VALUES ===============================================
DATE_LB="20100101"
DATE_UB="20100201"

# define location of files
GOSAT_FILE_DEST="/Users/mikestanley/Research/Carbon_Flux/data/modeled_satellite_obs/JULES_pert"
PERTURB_FILE_SOURCE="/Users/mikestanley/Research/Carbon_Flux/data/modeled_satellite_obs/pert_files"
UNPERT_GOSAT_SOURCE="/Users/mikestanley/Research/Carbon_Flux/data/modeled_satellite_obs/JULES_unpert"

# number of OSSEs to do
MAX_COUNT=10

# run path for OSSE of interest
RUN_PATH="/glade/u/home/mcstanley/pbs_run_scripts/run_reproduce_brendan_results"

# ============================================================================

# confirm that each of the above directories exists
if [ ! -d $GOSAT_FILE_DEST ] || [ ! -d $PERTURB_FILE_SOURCE ] || [ ! -d $UNPERT_GOSAT_SOURCE ];
then
    echo "Some directory location does not exist."
    exit 1
fi

# loop through data generation process for as many desired iterations
count=0
while [ $count -lt $MAX_COUNT ]
do

    echo "Generating GOSAT files (iter num: $count)"
    python ./generate_gosat_data.py \
        --base_file_dir="$UNPERT_GOSAT_SOURCE" \
        --date_lb="$DATE_LB"                   \
        --date_ub="$DATE_UB"                   \
        --output_dir="$GOSAT_FILE_DEST"        \
        --perturb_dir="$PERTURB_FILE_SOURCE"   \
        --perturb_idx="$count"

    #echo "Initiating GEOS-Chem Adjoint"
    #qsub $RUN_PATH
    # TODO need to find a way to keep track of pbs job ID
    
    # move the last scale factor file to save location

    count=$((count+1))
done

echo "Complete."