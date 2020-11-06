#!/bin/bash
#
# Given a directory of optimization results, find the last optimized one.
#
# Author        : Mike Stanley
# Created       : November 4, 2020
# Last Modified : November 4, 2020
# =============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo
MULTIJOB_DIR=$BASE_DIR/multijob
ENS_STEM=runs/v8-02-01/geos5/OptData

# make a directory for the raw results
mkdir -p $DEST_DIR/OptData

# create a file to store the counts
touch ./count_file_2point25.txt

count=0
for path in $MULTIJOB_DIR/ens*
do

    max_val=0
    for sf_file in $path/$ENS_STEM/gctm.sf.*
    do
        opt_num=$(echo $sf_file | rev | cut -c1-2 | rev)

        if [[ $opt_num -gt $max_val ]]; then
            max_val=$opt_num
        fi
    done
    echo "Ens_${count} = ${max_val}" >> ./count_file.txt

    count=$((count + 1))
done