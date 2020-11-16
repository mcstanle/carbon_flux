#!/bin/bash
# Launches a collection of cmdfile's with qsub
#
# Author        : Mike Stanley
# Created       : October 28, 2020
# Last Modified : November 16, 2020
#==============================================================================
BASE_DIR=/glade/work/mcstanley/monte_carlo/multijob
FILE_NMS=$BASE_DIR/launch_file*

for file in $FILE_NMS
do
    qsub $file
done
