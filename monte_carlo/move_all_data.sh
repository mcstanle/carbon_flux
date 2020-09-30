#!/bin/bash
#
# Script to move all optimization data into a new location
#
# =============================================================================

BASE_DIR=/glade/work/mcstanley/monte_carlo/data_archive
monte_carlo_run_num=1  # which overall run is being saved?
ORIG_DIR=$BASE_DIR/V1
NEW_DIR=$BASE_DIR/run_$monte_carlo_run_num

mkdir NEW_DIR

# copy the results directory
cp -r $ORIG_DIR/results $NEW_DIR

# copy each OptData
#