#!/bin/bash
#
# Move the raw optimized data from the multijob monte carlo run to data archive.
#
# Also moves the stderr and stdout
#
# Author        : Mike Stanley
# Created       : November 3, 2020
# Last Modified : November 9, 2020
# =============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo
MULTIJOB_DIR=$BASE_DIR/multijob
ENS_STEM=runs/v8-02-01/geos5/OptData
FILE_STEM=runs/v8-02-01/geos5
DEST_DIR=$BASE_DIR/data_archive/run_5

# make a directory for the raw results
mkdir -p $DEST_DIR/OptData

# make a directory for the stdout and stderr files
mkdir -p $DEST_DIR/output_files
mkdir -p $DEST_DIR/output_files/stdout_files
mkdir -p $DEST_DIR/output_files/stderr_files

count=0
for path in $MULTIJOB_DIR/ens*
do

    # MOVING DATA
    # make a sub directory
    mkdir -p $DEST_DIR/OptData/element_$count

    cp -r $path/$ENS_STEM/* $DEST_DIR/OptData/element_$count

    # MOVE STDOUT AND STDERR
    cp $path/$FILE_STEM/stdout_$count.txt $DEST_DIR/output_files/stdout_files
    cp $path/$FILE_STEM/stderr_$count.txt $DEST_DIR/output_files/stderr_files

    echo "Element ${count}"
    count=$((count + 1))
done
