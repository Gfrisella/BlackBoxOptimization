#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`readlink -f .`
#source $PROJECTS_DIR/cluster/set_vars.sh

cd ~/Project/MuonsAndMatter
export PYTHONPATH=$PYTHONPATH:`readlink -f python`:`readlink -f cpp/build`

python3 ~/Project/BlackBoxOptimization/src/problems_dev_mine.py
#python3 /mnt/BlackBoxOptimization/run_optimization.py --problem 'stochastic_rosenbrock'
