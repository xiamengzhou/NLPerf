#!/bin/sh
#SBATCH --mem=24g
#SBATCH -t 0

CODE=$data2/software/nlppred

task=$1
n=$2

python3 $CODE/src/new_model.py \
        --task $task \
        --log ../logs/${task}_nm_${n}.log \
        --portion 0.5 \
        --test_id_options_num 100 \
        --sample_options_num 100



