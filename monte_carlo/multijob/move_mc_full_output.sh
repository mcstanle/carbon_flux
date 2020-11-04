#!/bin/bash
#
# Move the raw optimized data from the multijob monte carlo run to data archive.
#
# Author        : Mike Stanley
# Created       : November 3, 2020
# Last Modified : November 3, 2020
# =============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo
MULTIJOB_DIR=$BASE_DIR/multijob
ENS_STEM=runs/v8-02-01/geos5/OptData
DEST_DIR=$BASE_DIR/data_archive/run_2

# make a directory for the raw results
mkdir -p $DEST_DIR/OptData

count=0
for path in $MULTIJOB_DIR/ens*
do

    # make a sub directory
    mkdir -p $DEST_DIR/OptData/element_$count

    cp -r $path/$ENS_STEM/* $DEST_DIR/OptData/element_$count

    count=$((count + 1))
done