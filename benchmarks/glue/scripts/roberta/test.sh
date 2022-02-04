#!/bin/bash -
#===============================================================================
#
#          FILE: test.sh
#
#         USAGE: ./test.sh
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
unitorch-auto-infer glue/cola/roberta.ini --from_ckpt_dir ./cola/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/CoLA.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/mnli_matched/roberta.ini --from_ckpt_dir ./mnli_matched/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/MNLI-m.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/mnli_mismatched/roberta.ini --from_ckpt_dir ./mnli_mismatched/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/MNLI-mm.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/mrpc/roberta.ini --from_ckpt_dir ./mrpc/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/MRPC.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/qnli/roberta.ini --from_ckpt_dir ./qnli/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/QNLI.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/qqp/roberta.ini --from_ckpt_dir ./qqp/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/QQP.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/rte/roberta.ini --from_ckpt_dir ./rte/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/RTE.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/sst2/roberta.ini --from_ckpt_dir ./sst2/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/SST-2.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/stsb/roberta.ini --from_ckpt_dir ./stsb/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/STS-B.tsv
unitorch-auto-infer glue/wnli/roberta.ini --from_ckpt_dir ./wnli/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/WNLI.tsv --core/process/general@act_fn softmax
unitorch-auto-infer glue/ax/roberta.ini --from_ckpt_dir ./mnli_matched/roberta/$SEED --core/task/supervised_task@output_path test/roberta/$SEED/AX.tsv --core/process/general@act_fn softmax

