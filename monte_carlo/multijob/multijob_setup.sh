#!/bin/bash
#
# Script to set up a multijob directory
#
# 1. Creates cmdfile (list of file paths to the element bash scripts)
# 2. Creates each bash script in the above file
# 3. Creates directory for each element in the above
#
# INPUTS
# 1. Target directory where files and directories can be made
# 2. Template element bash script
# 3. Pre-compiled GEOS-Chem directory
#
# NOTE:
# - the sample element bash script is expected to be in the same directory
#   as this script
#
# STEPS FOR USE
# 1. set directory where multijob will be executed (BASE_DIR)
# 2. set directory where compiled code lives (COMPILED_CODE_DIR)
# 3. define number of ensemble elements (NUM_ELEMENTS)
#
# Author        : Mike Stanley
# Created       : October 27, 2020
# Last Modified : October 27, 2020
#==============================================================================

# operational params
BASE_DIR=/glade/work/mcstanley/monte_carlo/multijob_test
COMPILED_CODE_DIR=/glade/u/home/mcstanley/gc_adj_runs/monte_carlo_element
NUM_ELEMENTS=4

# stems
SCRIPT_STEM=element
DIR_STEP=ens

# create and fill the cmdfile
touch $BASE_DIR/cmdfile

for i in $(seq 0 $( expr ${NUM_ELEMENTS} - 1 ))
do
    echo $BASE_DIR/${SCRIPT_STEM}_${i}.sh >> $BASE_DIR/cmdfile
done

# copy sample bash script and alter element number
for i in $(seq 0 $( expr ${NUM_ELEMENTS} - 1 ))
do
    # create the copy for element i
    cp ./sample_element.sh $BASE_DIR/${SCRIPT_STEM}_${i}.sh

    # alter the ENS_NUM= line
    # NOTE: there might be platform dependence with the sed command
    replace_str="ENS_NUM=${i}"
    sed -i.bu "s/ENS_NUM=0/${replace_str}/" $BASE_DIR/${SCRIPT_STEM}_${i}.sh

    # remove the created backup
    rm $BASE_DIR/${SCRIPT_STEM}_${i}.sh.bu

done

# create the code directories for each element
for i in $(seq 0 $( expr ${NUM_ELEMENTS} - 1 ))
do
    # make the directory by copying over contents
    cp -r $COMPILED_CODE_DIR/* $BASE_DIR/ens_${i}

done