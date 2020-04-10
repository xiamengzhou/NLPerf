#!/bin/sh
#SBATCH -t 0
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --job-name="collect_feats"

# python3 $nlppred/collect_feats.py
SRC=en
TGT=tr
LANG=${SRC}_${TGT}
DATA=$data2/wikimatrix/spm5k/${LANG}

source activate allennlp
types=(10k 50k 100k 200k all)
# shellcheck disable=SC2068
for type in ${types[@]}; do
  BIN_DIR=$DATA/bins/$LANG/$type
  save_dir=$out/${LANG}2/$type
  fairseq-generate $BIN_DIR \
                  --source-lang ${SRC} \
                  --target-lang ${TGT} \
                  --path $save_dir/checkpoint_best.pt \
                  --beam 5 \
                  --nbest 1 \
                  --remove-bpe=sentencepiece \
                  --lenpen 1.2 \
                  --batch-size 100 \
                  --sacrebleu \
                  --gen-subset test > $save_dir/eval_log
done
