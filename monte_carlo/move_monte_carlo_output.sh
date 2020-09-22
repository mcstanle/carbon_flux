#!/bin/bash
#
# Automatically moves the following important pieces of information from the
# silo'ed results into one location.
#   - last scale factor iteration (must be specified)
#   - tracerinfo and diaginfo files
#
# NOTES:
#   - the NUM_ITERATIONS variable must be left padded with a zero if <10
#
#
# Author        : Mike Stanley
# Created       : September 22, 2020
# Last Modified : September 22, 2020
# =============================================================================

# operational parameters
NUM_ELEMENTS=5
NUM_ITERATIONS=02   # this tells us which scale factor files to search for

# location information
BASE_DIR=/glade/work/mcstanley/monte_carlo/V1
BASE_SF_LOC=monte_carlo_element/runs/v8-02-01/geos5/OptData
DEST_DIR=$BASE_DIR/results

# create a directory to store the results
mkdir -p $DEST_DIR

# obtain the optimized scale factors
for i in $(seq 0 $( expr ${NUM_ELEMENTS} - 1))
do

    cp $BASE_DIR/src/element_${i}/$BASE_SF_LOC/gctm.sf.$NUM_ITERATIONS $DEST_DIR/opt_sf_$i

done

# grab the tracer and diag info from the first element
cp $BASE_DIR/src/element_0/monte_carlo_element/runs/v8-02-01/geos5/tracerinfo.dat $DEST_DIR/
cp $BASE_DIR/src/element_0/monte_carlo_element/runs/v8-02-01/geos5/diaginfo.dat $DEST_DIR/