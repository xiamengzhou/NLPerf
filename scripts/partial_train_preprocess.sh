#!/bin/sh
#SBATCH --mem=24g
#SBATCH -t 0

source activate allennlp

sample() {
    # sample data from existing data
  suffixes=(10k 50k 100k 200k 500k 1000k 1500k 2000k)
  lines=(10000 50000 100000 200000 500000 1000000 1500000 2000000)


  for i in `seq 0 $((${#suffixes[@]}-1))`; do
    suffix=${suffixes[$i]}
    line=${lines[$i]}
    new_src_file=$DATA/WikiMatrix.en-${TGT}.txt.${SRC}.spm5k.$suffix
    new_tgt_file=$DATA/WikiMatrix.en-${TGT}.txt.${TGT}.spm5k.$suffix
    if [ ! -f $new_src_file ] || [ ! -f $new_tgt_file ]; then
      python3 $UTIL sample_para $SRC_FILE $TGT_FILE \
              $new_src_file \
              $new_tgt_file \
              $line
    fi
  done
}

spm_dev_test_data() {
  # prepare dev test set
  SRC_ISO3=$1
  TGT_ISO3=$2
  TOK_DIR=$DATA
  DEV_SOURCE=$TOK_DIR/dev.${SRC_ISO3}
  DEV_TARGET=$TOK_DIR/dev.${TGT_ISO3}
  TEST_SOURCE=$TOK_DIR/test.${SRC_ISO3}
  TEST_TARGET=$TOK_DIR/test.${TGT_ISO3}
  SPM_DIR=$data4/wikimatrix/spm-model
  SRC_SPM_MODEL=$SPM_DIR/WikiMatrix.${SRC}-${TGT}.txt.${SRC}.tok.model
  TGT_SPM_MODEL=$SPM_DIR/WikiMatrix.${SRC}-${TGT}.txt.${TGT}.tok.model
  OUTPUT_DEV_SOURCE=$DATA/ted-dev.spm5k.${SRC}-${TGT}.${SRC}
  OUTPUT_DEV_TARGET=$DATA/ted-dev.spm5k.${SRC}-${TGT}.${TGT}
  OUTPUT_TEST_SOURCE=$DATA/ted-test.spm5k.${SRC}-${TGT}.${SRC}
  OUTPUT_TEST_TARGET=$DATA/ted-test.spm5k.${SRC}-${TGT}.${TGT}
  python3 $SPM encode $SRC_SPM_MODEL $DEV_SOURCE $OUTPUT_DEV_SOURCE
  python3 $SPM encode $SRC_SPM_MODEL $TEST_SOURCE $OUTPUT_TEST_SOURCE
  python3 $SPM encode $TGT_SPM_MODEL $DEV_TARGET $OUTPUT_DEV_TARGET
  python3 $SPM encode $TGT_SPM_MODEL $TEST_TARGET $OUTPUT_TEST_TARGET
}


move_file() {
  suffixes=(10k 50k 100k 200k 500k 1000k 1500k 2000k all)
  for i in `seq 0 $((${#suffixes[@]}-1))`; do
    suffix=${suffixes[$i]}
    files=$(ls $DATA/*${suffix})
    if [[ ! -z ${files} ]]
      then
        new_src_file=$DATA/WikiMatrix.${SRC}-${TGT}.txt.${SRC}.spm5k.$suffix
        new_tgt_file=$DATA/WikiMatrix.${SRC}-${TGT}.txt.${TGT}.spm5k.$suffix
        mv $new_src_file $DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix.${SRC}
        mv $new_tgt_file $DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix.${TGT}
    fi
  done
  new_src_file=$DATA/WikiMatrix.${SRC}-${TGT}.txt.${SRC}.spm5k
  new_tgt_file=$DATA/WikiMatrix.${SRC}-${TGT}.txt.${TGT}.spm5k
  mv $new_src_file $DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.${SRC}
  mv $new_tgt_file $DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.${TGT}
}

make_bins() {
  suffixes=(10k 50k 100k 200k 500k 1000k 1500k 2000k)
  for i in `seq 0 $((${#suffixes[@]}-1))`; do
    # get the suffix
    suffix=${suffixes[$i]}

    files=$(ls $DATA/*${suffix}*)

    if [[ ! -z ${files} ]]
      then
        # create the destination dir
        DEST_DIR=$DATA/bins/$LANG/$suffix
        mkdir -p $DEST_DIR

        # preprocess
        fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
                           --trainpref $DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix \
                           --validpref $DATA/ted-dev.spm5k.${SRC}-${TGT} \
                           --testpref $DATA/ted-test.spm5k.${SRC}-${TGT} \
                           --destdir ${DEST_DIR} \
                           --joined-dictionary

        # echo after preprocess is done
        echo $LANG $suffix done!
    fi
  done

#  DEST_DIR=$DATA/bins/$LANG/all
#  mkdir -p $DEST_DIR
#  fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
#                       --trainpref $DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k \
#                       --validpref $DATA/ted-dev.spm5k.${SRC}-${TGT} \
#                       --testpref $DATA/ted-test.spm5k.${SRC}-${TGT} \
#                       --destdir ${DEST_DIR} \
#                       --joined-dictionary
}

debpe() {
  suffixes=(10k 50k 100k 200k 500k 1000k 1500k 2000k all)
  SPM_DIR=$data4/wikimatrix/spm-model
  SRC_SPM_MODEL=$SPM_DIR/WikiMatrix.${SRC}-${TGT}.txt.${SRC}.tok.model
  TGT_SPM_MODEL=$SPM_DIR/WikiMatrix.${SRC}-${TGT}.txt.${TGT}.tok.model
  for i in `seq 0 $((${#suffixes[@]}-1))`; do
    suffix=${suffixes[$i]}
    SRC_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix.${SRC}
    TGT_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix.${TGT}
    SRC_TOK_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.tok.$suffix.${SRC}
    TGT_TOK_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.tok.$suffix.${TGT}
    if [ -f $SRC_FILE ]; then
      spm_decode --model $SRC_SPM_MODEL --output $SRC_TOK_FILE $SRC_FILE
      spm_decode --model $TGT_SPM_MODEL --output $TGT_TOK_FILE $TGT_FILE
    fi
  done
}

# get frequency table
vocab() {
  tr ' ' '\n' < "$1" \
 | sort \
 | uniq -c \
 | sort -rn \
 | awk '{ printf "%s %s\n", $2, $1 }' > $2
}

get_vocab() {
  suffixes=(10k 50k 100k 200k 500k 1000k 1500k 2000k all)
  SPM_DIR=$data4/wikimatrix/spm-model
  for i in `seq 0 $((${#suffixes[@]}-1))`; do
    suffix=${suffixes[$i]}
    SRC_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix.${SRC}
    TGT_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.spm5k.$suffix.${TGT}
    SRC_TOK_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.tok.$suffix.${SRC}
    TGT_TOK_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.tok.$suffix.${TGT}
#    if [-f $SRC_FILE ]; then
    # vocab $SRC_FILE ${SRC_FILE}.vocab
    # vocab $TGT_FILE ${TGT_FILE}.vocab
#    fi
    if [ -f $SRC_TOK_FILE ]; then
      vocab $SRC_TOK_FILE ${SRC_TOK_FILE}.vocab
      vocab $TGT_TOK_FILE ${TGT_TOK_FILE}.vocab
    fi
  done
}

delete_empty_file() {
   find $1 -size 0 -delete
}

collect_feats() {
  declare -A lc=(["ar"]="ara" ["cs"]="ces" ["ko"]="kor" ["bs"]="bos" ["sr"]="srp" ["da"]="dan" ["de"]="deu"
                 ["he"]="heb" ["el"]="ell" ["eo"]="epo" ["hu"]="hun" ["en"]="eng" ["pl"]="pol" ["fr"]="fra"
                 ["pt"]="por" ["tr"]="tur")
  echo $SRC
  echo $TGT
  echo ${lc[$SRC]}
  echo ${lc[$TGT]}
  CODE_DIR="/projects/tir2/users/mengzhox/data/software/nlppred"
  python3 $CODE_DIR/src/preprocess/collect_feats.py process_multiple  --src=$SRC --tgt=$TGT --src3=${lc[$SRC]} --tgt3=${lc[$TGT]}
}


#SRC=en
#TGT=pt
#LANG=${SRC}_${TGT}
#DATA=$data4/wikimatrix/spm5k/${LANG}
#SRC_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.${SRC}.spm5k
#TGT_FILE=$DATA/WikiMatrix.${SRC}-${TGT}.txt.${TGT}.spm5k
#SPM=/home/mengzhox/Scripts/py_scripts/sentence_piece.py
#UTIL=/home/mengzhox/Scripts/py_scripts/utils.py
#

readarray pairs < /projects/tir4/users/mengzhox/wikimatrix/spm5k/run_pairs

for i in `seq 0 $((${#pairs[@]}-1))`; do
  pair=${pairs[$i]}
  set -- $(echo $pair | tr "_" "\n")
  SRC=$1
  TGT=$2
  LANG=${SRC}_${TGT}
  DATA=$data4/wikimatrix/spm5k/${LANG}
  collect_feats
done

