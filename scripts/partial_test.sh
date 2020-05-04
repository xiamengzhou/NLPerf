#!/bin/sh
#SBATCH -t 0
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --job-name="test"

readarray pairs < /projects/tir4/users/mengzhox/wikimatrix/spm5k/pairs

source activate allennlp
types=(1500k 2000k all)
pairs=('pt en')

# shellcheck disable=SC2068
for ((i=0; i<${#pairs[@]}; i++)) ; do
pair=${pairs[$i]}
set -- $(echo $pair | tr " "  "\n")
SRC=$1
TGT=$2

if [[ "$SRC" > "$TGT" ]]; then
  LANG=${TGT}_${SRC}
else
  LANG=${SRC}_${TGT}
fi

DATA=$data4/wikimatrix/spm5k/${LANG}
OUT=$data4/wikimatrix/spm5k/out/${SRC}_${TGT}

for type in ${types[@]}; do
  save_dir=$OUT/$type
  echo $save_dir
  if [ -d $save_dir ]; then
    echo hey
    echo Decoding ${SRC}_${TGT} $type...
    BIN_DIR=$DATA/bins/$LANG/$type
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
     echo Decoding ${SRC}_${TGT} $type done!
  fi
done
done
