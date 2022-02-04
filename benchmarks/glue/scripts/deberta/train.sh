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
unitorch-auto-train glue/cola/deberta.ini --cache_dir ./cola/deberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-auto-train glue/mnli_matched/deberta.ini --cache_dir ./mnli_matched/deberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-auto-train glue/mnli_mismatched/deberta.ini --cache_dir ./mnli_mismatched/deberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-auto-train glue/mrpc/deberta.ini --cache_dir ./mrpc/deberta/$SEED --core/task/supervised_task@seed $SEED --core/model/classification/deberta@pretrained_weight_path ./mnli_matched/deberta/$SEED/pytorch_model.bin
unitorch-auto-train glue/qnli/deberta.ini --cache_dir ./qnli/deberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-auto-train glue/qqp/deberta.ini --cache_dir ./qqp/deberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-auto-train glue/rte/deberta.ini --cache_dir ./rte/deberta/$SEED --core/task/supervised_task@seed $SEED --core/model/classification/deberta@pretrained_weight_path ./mnli_matched/deberta/$SEED/pytorch_model.bin
unitorch-auto-train glue/sst2/deberta.ini --cache_dir ./sst2/deberta/$SEED --core/task/supervised_task@seed $SEED
unitorch-auto-train glue/stsb/deberta.ini --cache_dir ./stsb/deberta/$SEED --core/task/supervised_task@seed $SEED --core/model/classification/deberta@pretrained_weight_path ./mnli_matched/deberta/$SEED/pytorch_model.bin
unitorch-auto-train glue/wnli/deberta.ini --cache_dir ./wnli/deberta/$SEED --core/task/supervised_task@seed $SEED

