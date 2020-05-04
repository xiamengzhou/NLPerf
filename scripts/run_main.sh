#!/bin/sh
#SBATCH --mem=24g
#SBATCH -t 0

CODE=$data2/software/nlppred

python3 $CODE/src/main_code.py
