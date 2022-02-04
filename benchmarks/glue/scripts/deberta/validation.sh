#!/bin/bash -
#===============================================================================
#
#          FILE: validation.sh
#
#         USAGE: ./validation.sh
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
unitorch-auto-infer glue/cola/deberta.ini --from_ckpt_dir ./cola/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/CoLA.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-auto-infer glue/mnli_matched/deberta.ini --from_ckpt_dir ./mnli_matched/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/MNLI-m.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation_matched
unitorch-auto-infer glue/mnli_mismatched/deberta.ini --from_ckpt_dir ./mnli_mismatched/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/MNLI-mm.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation_mismatched
unitorch-auto-infer glue/mrpc/deberta.ini --from_ckpt_dir ./mrpc/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/MRPC.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-auto-infer glue/qnli/deberta.ini --from_ckpt_dir ./qnli/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/QNLI.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-auto-infer glue/qqp/deberta.ini --from_ckpt_dir ./qqp/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/QQP.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-auto-infer glue/rte/deberta.ini --from_ckpt_dir ./rte/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/RTE.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-auto-infer glue/sst2/deberta.ini --from_ckpt_dir ./sst2/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/SST-2.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-auto-infer glue/stsb/deberta.ini --from_ckpt_dir ./stsb/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/STS-B.tsv --core/dataset/ast/test@split validation
unitorch-auto-infer glue/wnli/deberta.ini --from_ckpt_dir ./wnli/deberta/$SEED --core/task/supervised_task@output_path validation/deberta/$SEED/WNLI.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation

