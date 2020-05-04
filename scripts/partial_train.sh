#!/bin/sh
#SBATCH -t 0
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --job-name="train"

SRC=$1
TGT=$2

if [[ "$SRC" > "$TGT" ]]; then
  LANG=${TGT}_${SRC}
else
  LANG=${SRC}_${TGT}
fi

DATA=$data4/wikimatrix/spm5k/${LANG}

source activate allennlp
types=($3)
# shellcheck disable=SC2068
for type in ${types[@]}; do
  BIN_DIR=$DATA/bins/${LANG}/$type
  save_dir=$data4/wikimatrix/spm5k/out/${SRC}_${TGT}/$type
  mkdir -p $save_dir
  echo $save_dir
  fairseq-train $BIN_DIR \
            --source-lang ${SRC} --target-lang ${TGT} \
            --arch transformer \
            --share-all-embeddings \
            --encoder-layers 5 \
            --decoder-layers 5 \
            --encoder-embed-dim 512 \
            --decoder-embed-dim 512 \
            --encoder-ffn-embed-dim 2048 \
            --decoder-ffn-embed-dim 2048 \
            --encoder-attention-heads 2 \
            --decoder-attention-heads 2 \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --dropout 0.4 \
            --attention-dropout 0.2 \
            --relu-dropout 0.2 \
            --weight-decay 0.0001 \
            --label-smoothing 0.2 \
            --criterion label_smoothed_cross_entropy \
            --optimizer adam \
            --adam-betas '(0.9, 0.98)' \
            --clip-norm 0 \
            --lr-scheduler inverse_sqrt \
            --warmup-updates 4000 \
            --warmup-init-lr 1e-7 \
            --lr 1e-3 --min-lr 1e-9 \
            --max-tokens 2000 \
            --update-freq 8 \
            --max-epoch 100 \
            --save-dir $save_dir \
            --save-interval 10 > $save_dir/log
done

