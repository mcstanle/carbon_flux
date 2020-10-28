#!/bin/bash
# Launches a collection of cmdfile's with qsub
#
# Author        : Mike Stanley
# Created       : September 21, 2020
# Last Modified : October 27, 2020
#==============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo/multijob_test
FILE_NMS=$BASE_DIR/cmdfile*

for file in $FILE_NMS
do
    qsub $file
done
