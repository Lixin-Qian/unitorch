#!/bin/bash -
#===============================================================================
#
#          FILE: train.sh
#
#         USAGE: ./train.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: fuliucansheng (fuliu), fuliucansheng@gmail.com
#  ORGANIZATION:
#       CREATED: 12/19/2021 00:49
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

SEED=$1
unitorch-train benchmarks/glue/cola/roberta.ini --cache_dir ./cola/roberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-train benchmarks/glue/mnli_matched/roberta.ini --cache_dir ./mnli_matched/roberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-train benchmarks/glue/mnli_mismatched/roberta.ini --cache_dir ./mnli_mismatched/roberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-train benchmarks/glue/mrpc/roberta.ini --cache_dir ./mrpc/roberta/$SEED --core/task/supervised_task@seed $SEED --core/model/classification/roberta@pretrained_weight_path ./mnli_matched/roberta/$SEED/pytorch_model.bin
unitorch-train benchmarks/glue/qnli/roberta.ini --cache_dir ./qnli/roberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-train benchmarks/glue/qqp/roberta.ini --cache_dir ./qqp/roberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-train benchmarks/glue/rte/roberta.ini --cache_dir ./rte/roberta/$SEED --core/task/supervised_task@seed $SEED --core/model/classification/roberta@pretrained_weight_path ./mnli_matched/roberta/$SEED/pytorch_model.bin
unitorch-train benchmarks/glue/sst2/roberta.ini --cache_dir ./sst2/roberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-train benchmarks/glue/stsb/roberta.ini --cache_dir ./stsb/roberta/$SEED --core/task/supervised_task@seed $SEED --core/model/classification/roberta@pretrained_weight_path ./mnli_matched/roberta/$SEED/pytorch_model.bin
unitorch-train benchmarks/glue/wnli/roberta.ini --cache_dir ./wnli/roberta/$SEED --core/task/supervised_task@seed $SEED

