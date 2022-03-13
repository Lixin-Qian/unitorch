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
unitorch-infer benchmarks/glue/cola/roberta.ini --from_ckpt_dir ./cola/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/CoLA.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/mnli_matched/roberta.ini --from_ckpt_dir ./mnli_matched/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/MNLI-m.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation_matched
unitorch-infer benchmarks/glue/mnli_mismatched/roberta.ini --from_ckpt_dir ./mnli_mismatched/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/MNLI-mm.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation_mismatched
unitorch-infer benchmarks/glue/mrpc/roberta.ini --from_ckpt_dir ./mrpc/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/MRPC.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/qnli/roberta.ini --from_ckpt_dir ./qnli/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/QNLI.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/qqp/roberta.ini --from_ckpt_dir ./qqp/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/QQP.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/rte/roberta.ini --from_ckpt_dir ./rte/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/RTE.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/sst2/roberta.ini --from_ckpt_dir ./sst2/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/SST-2.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/stsb/roberta.ini --from_ckpt_dir ./stsb/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/STS-B.tsv --core/dataset/ast/test@split validation
unitorch-infer benchmarks/glue/wnli/roberta.ini --from_ckpt_dir ./wnli/roberta/$SEED --core/task/supervised_task@output_path validation/roberta/$SEED/WNLI.tsv --core/process/general@act_fn softmax --core/dataset/ast/test@split validation

