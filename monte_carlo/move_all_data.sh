#!/bin/bash
#
# Script to move all optimization data into a new location
#
# =============================================================================

BASE_DIR=/glade/work/mcstanley/monte_carlo
monte_carlo_run_num=1  # which overall run is being saved?
ORIG_DIR=$BASE_DIR/V1
NEW_DIR=$BASE_DIR/data_archive/run_$monte_carlo_run_num
OPT_DATA_SUB_DIR=monte_carlo_element/runs/v8-02-01/geos5/OptData

mkdir $NEW_DIR

# copy the results directory
cp -r $ORIG_DIR/results $NEW_DIR

# copy each OptData
for file in $(ls $ORIG_DIR/src/element_*); do
    echo $file/$OPT_DATA_SUB_DIR
done