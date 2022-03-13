
## GLUE Benchmarks

> Go To [Offical Leaderboard](https://gluebenchmark.com/leaderboard).

#### Typical workflow for fine-tuning to SOTA on GLUE
* Pre-train your model with as much data/compute as possible
* Tune fine-tuning hyperparameters on the dev sets
* Use the SuperGLUE rather than GLUE data for WNLI and implement rescoring trick in combination with using additional labeled (“Definite Pronoun Resolution Dataset” http://www.hlt.utdallas.edu/~vince/data/emnlp12/) or unlabeled data (Vid Kocijan et al., “A Surprisingly Robust Trick for Winograd Schema Challenge,” ACL 2019)
* Use a special (and not officially allowed) pairwise ranking trick for QNLI and WNLI (users are not supposed to share information across test examples)
* Intermediate MNLI task fine-tuning for MRPC/STS/RTE
* Fine-tune many models on each task. Ensemble the best 5-10 models for each task.
* Submit a (single) final run to the test leaderboard


#### Valiadation Performance

| #   | name                  | cola | mnli_matched | mnli_mismatched | mrpc | qnli | qqp  | rte  | sst2 | stsb | wnli | ax   |
|:---:|:---------------------:|:----:|:------------:|:---------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 0   |roberta-large          |      |              |                 |      |      |      |      |      |      |      |      |
| 1   |deberta-large          |      |              |                 |      |      |      |      |      |      |      |      |

