#!/bin/sh
#SBATCH --mem=24g
#SBATCH -t 0

CODE=$data2/software/nlppred

task=$1
ttype=$2
n=5
beam_size=100

declare -A arr
arr["best_search"]=bs
arr["worst_search"]=ws
arr["random_search"]=rs

echo $task
python3 $CODE/src/representativeness.py \
        --task $task \
        --n $n \
        --beam_size $beam_size \
        --log ../logs/${task}_${arr[$ttype]}.log \
        --type $ttype


# run in bash
# for task in monomt wiki tsfmt tsfparsing tsfpos tsfel bli ma ud; do
#   for ttype in best_search worst_search random_search; do
#     sbatch run_rep.sh $task $ttype
#  done
# done