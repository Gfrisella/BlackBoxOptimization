#!/bin/bash

#SBATCH --job-name=simulations
#SBATCH --output=slurm_%j.out       # Standard output and error log

# ssh linux.physik.uzh.ch 'echo "no" >> dontwork'

export PROJECTS_DIR=~/Project
export STORE_DIR=~
export PYTHONNOUSERSITE=1
export APPTAINER_DIR=~/Project/container/

# Check if the required directories exist
if [ ! -d "$PROJECTS_DIR" ]; then
    echo "Error: Directory $PROJECTS_DIR does not exist!"
    exit 1
fi

if [ ! -d "$STORE_DIR" ]; then
    echo "Error: Directory $STORE_DIR does not exist!"
    exit 1
fi

# Check if there are any files to remove
if ls $STORE_DIR/Project/BlackBoxOptimization/Outputs/results_rank_* 1> /dev/null 2>&1; then
    # If files exist, remove them
    echo "Removing files matching 'results_rank_*'..."
    rm $STORE_DIR/Project/BlackBoxOptimization/Outputs/results_rank_*
    echo "Files removed successfully."
else
    echo "No 'results_rank_*' files found to remove."
fi


# Check if there are any files to remove
if ls $STORE_DIR/Project/BlackBoxOptimization/Outputs/merged_data* 1> /dev/null 2>&1; then
    # If files exist, remove them
    echo "Removing files matching 'merged_data.pkl'..."
    rm $STORE_DIR/Project/BlackBoxOptimization/Outputs/merged_data*
    echo "Files removed successfully."
else
    echo "No 'merged_data.pkl' files found to remove."
fi
# Run the container with bind-mounts
srun -A uzh42 -C mc bash apptainer exec -B ~ -B "APPTAINER_DIR"-B "$STORE_DIR"   "~/Project/container/snoopy_geant.sif" / 
bash ~/Project/BlackBoxOptimization/run_worker.sh

# bash "$STORE_DIR/Project/container/install-dir/bin/apptainer" exec \
#     -B /cvmfs \
#     -B "$STORE_DIR" \
#     -B "/home/hep/gfrise/Project:/mnt" \
#     "/disk/users/lprate/containers/fem_geant.sif" \
#     python3 $STORE_DIR/Project/BlackBoxOptimization/src/Merging_results.py --rm
