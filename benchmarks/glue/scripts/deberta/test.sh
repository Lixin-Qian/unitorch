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
unitorch-infer benchmarks/glue/cola/deberta.ini --from_ckpt_dir ./cola/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/CoLA.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/mnli_matched/deberta.ini --from_ckpt_dir ./mnli_matched/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/MNLI-m.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/mnli_mismatched/deberta.ini --from_ckpt_dir ./mnli_mismatched/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/MNLI-mm.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/mrpc/deberta.ini --from_ckpt_dir ./mrpc/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/MRPC.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/qnli/deberta.ini --from_ckpt_dir ./qnli/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/QNLI.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/qqp/deberta.ini --from_ckpt_dir ./qqp/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/QQP.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/rte/deberta.ini --from_ckpt_dir ./rte/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/RTE.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/sst2/deberta.ini --from_ckpt_dir ./sst2/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/SST-2.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/stsb/deberta.ini --from_ckpt_dir ./stsb/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/STS-B.tsv
unitorch-infer benchmarks/glue/wnli/deberta.ini --from_ckpt_dir ./wnli/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/WNLI.tsv --core/process/general@act_fn softmax
unitorch-infer benchmarks/glue/ax/deberta.ini --from_ckpt_dir ./mnli_matched/deberta/$SEED --core/task/supervised_task@output_path test/deberta/$SEED/AX.tsv --core/process/general@act_fn softmax

