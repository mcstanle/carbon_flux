#!/bin/bash
# this script is meant to make getting scale factor files from cheyenne easy
# CL arguments are:
#   --run_name
#   --month
#
# NOTE: expects the scale factor files to be in /glade/scratch/mcstanley/run_output/

# Cheyenne Constants
USRMN="mcstanley"
CHEYENNE_EXT="cheyenne.ucar.edu"
CHEYENNE_LOC="/glade/scratch/mcstanley/run_output/"
LOCAL_LOC="/Users/mikestanley/Research/Carbon_Flux/gc_adj_runs/"

# set default empty RUN_NAME and MONTH parameters
RUN_NAME="EMPTY"
MONTH="EMPTY"

# collect the commandline arguments
for i in "$@"
do
case $i in
    --run_name=*)
    RUN_NAME="${i#*=}"
    shift
    ;;
    --month=*)
    MONTH="${i#*=}"
    shift
    ;;
    --help)
    echo "Please enter --run_name=VAL and --month=VAL"
esac
done

if [ $RUN_NAME == "EMPTY" ] || [ $MONTH == "EMPTY" ];
then
    echo "Please enter a --run_name and --month"
    exit 1
fi

# build the source location
SOURCE_LOC="${CHEYENNE_LOC}${RUN_NAME}/OptData/${MONTH}/gctm.sf*"

# build the destination location
DEST_LOC="${LOCAL_LOC}${RUN_NAME}/OptData/${MONTH}/"

# copy all scaling factor files from Cheyenne to Local
scp -r $USRMN@$CHEYENNE_EXT:$SOURCE_LOC $DEST_LOC
