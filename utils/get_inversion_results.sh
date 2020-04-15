# move monthly inversion results from cheyenne to local machine
# we assume that the first command line argument is the name of inversion
# note that I have not found a way to check if a directory exists on cheyenne
#
# for some reason, the scp doesn't work when executing a bash script, so instead,
# we make this script return a string

# source/destination directory locations
SOURCE_LOC=/glade/scratch/mcstanley/run_output
DEST_LOC=/Users/mikestanley/Research/Carbon_Flux/gc_adj_runs

# ssh parameters
USR_NM="mcstanley"
HOST_NM="cheyenne.ucar.edu"

# get the run name
RUN_NAME=$1

# check if the directory already exists on the local machine
# if [ -d $DEST_LOC/$RUN_NAME ]
# then
#     echo "$DEST_LOC/$RUN_NAME already exists"
#     exit 0
# fi

# transfer the files
echo "scp -r ${USR_NM}@${HOST_NM}:$SOURCE_LOC/$RUN_NAME $DEST_LOC/$RUN_NAME"

# transfer the tracer and diag files
echo "scp -r ${USR_NM}@${HOST_NM}:/glade/u/home/mcstanley/gc_adj_runs/$RUN_NAME/runs/v8-02-01/geos5/tracerinfo.dat $DEST_LOC/$RUN_NAME"
echo "scp -r ${USR_NM}@${HOST_NM}:/glade/u/home/mcstanley/gc_adj_runs/$RUN_NAME/runs/v8-02-01/geos5/diaginfo.dat $DEST_LOC/$RUN_NAME"