# template for comparing scale factors

# define inputs
NUM_UNCERT=74
OSSE1_PATH=/Users/mikestanley/Research/Carbon_Flux/gc_adj_runs/increased_prior_uncertainty_XX/${NUM_UNCERT}/jan_sept_${NUM_UNCERT}
OSSE2_PATH=/Users/mikestanley/Research/Carbon_Flux/gc_adj_runs/reproduce_brendan_results/OptData/jan_sept
OSSE1_NM=prior_uncert_${NUM_UNCERT}
OSSE2_NM=prior_uncert_44
RESULT_DIR=/Users/mikestanley/Research/Carbon_Flux/OSSE_comp/${OSSE1_NM}_vs_${OSSE2_NM}

# check if RESULT_DIR doesn't exist
if [ ! -d $RESULT_DIR ]
then
    mkdir $RESULT_DIR
fi

# execute the script
python scale_factor_comp.py \
    --osse1_path=${OSSE1_PATH} \
    --osse2_path=${OSSE2_PATH} \
    --result_dir=${RESULT_DIR} \
    --osse1_name=${OSSE1_NM} \
    --osse2_name=${OSSE2_NM}
