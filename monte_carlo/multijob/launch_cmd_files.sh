#!/bin/bash
# Launches a collection of cmdfile's with qsub
#
# Author        : Mike Stanley
# Created       : October 28, 2020
# Last Modified : October 28, 2020
#==============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo/multijob_test
FILE_NMS=$BASE_DIR/launch_file*

for file in $FILE_NMS
do
    echo $file
done
