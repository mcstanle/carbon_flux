#!/bin/bash
#
# Given a directory of optimization results, find the last optimized one.
#
# Author        : Mike Stanley
# Created       : November 4, 2020
# Last Modified : November 10, 2020
# =============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo
# MULTIJOB_DIR=$BASE_DIR/data_archive/run_3/OptData
MULTIJOB_DIR=$BASE_DIR/multijob
ENS_STEM=runs/v8-02-01/geos5/OptData

# make a directory for the raw results
# mkdir -p $DEST_DIR/OptData

# create a file to store the counts
FILENAME=count_file_3pu_sub.txt
touch ./$FILENAME

count=0
for path in $MULTIJOB_DIR/ens*
# for path in $MULTIJOB_DIR/element_*
do

    max_val=0
    for sf_file in $path/$ENS_STEM/gctm.sf.*
    # for sf_file in $path/gctm.sf.*
    do
        opt_num=$(echo $sf_file | rev | cut -c1-2 | rev)

        if [[ "${opt_num#0}" -gt "${max_val#0}" ]]; then
            max_val=$opt_num
        fi
    done
    echo "Ens_${count} = ${max_val}" >> ./$FILENAME

    count=$((count + 1))
done