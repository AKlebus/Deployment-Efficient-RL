#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --job-name="de-rl"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="de-rl.out"
#SBATCH --error="de-rl.err"
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN,END

# python det-disc-DERL-hyperparameter-tuning.py --dim 4
# python stoc-DERL-hyperparameter-tuning.py --dim 4
# python stoc-DERL-hyperparameter-tuning.py --dim 5
python stoc-DERL-hyperparameter-tuning.py --dim 7
python stoc-DERL-hyperparameter-tuning.py --dim 5
# python stoc-DERL-hyperparameter-tuning.py --dim 10

# python det-cont-DERL-hyperparameter-tuning.py 

# run command every 10 seconds
# watch -n 10 squeue -u USERNAME_HERE

