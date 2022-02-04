
## Roberta Text Classification

#### Train with command line
##### Step 1: Prepare Customize Dataset
Prepare tsv file as an example. Also support json/hub dataset etc. The tsv should have three columns as dataset settings below.

##### Step 2: Prepare config.ini File
> Take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/classification/roberta.ini) as a template.

***Model Setting***
> The options in [core/model/classification/reberta] are setting for roberta model.
 

```ini
[core/model/classification/roberta]
pretrained_name = roberta-base
```

***Dataset Setting***

> * The options in [core/dataset/ast/train] are setting for training data. 
> * `names` is the fields of dataset. For tsv files, it should be the header.
> * `preprocess_functions` is the preprocess functions to convert the raw data into tensors as model inputs.
> * These options settings for train/dev/test are independent with each other.


```ini
[core/dataset/ast/train]
data_files = ${core/cli:train_file}
names = ['query', 'doc', 'label']
preprocess_functions = ['core/process/roberta_classification(query, doc)', 'core/process/label(label)']
```

***Process Setting***

> These setting are used for the pre/post processing classes.

```ini
[core/process/roberta]
pretrained_name = roberta-base
max_seq_length = 24

[core/process/general]
map_dict = {'Fair': 1, 'Good': 2, 'Bad': 0, 'Excellent': 3} # label mapping for label column.
num_class = 4
```

***Task Setting***
> * `loss` loss to be used in task.
> * `score_fn` Metric to be used for save checkpoint.
> * `monitor_fns` Metrics for logging.
> * `output_header` Save these fields for inference.
> * `post_process_fn`, `writer_fn` Post process & writer function.
> * `{train, dev, test}_batch_size` Batch size setting for train/eval/inference.

```ini
[core/task/supervised_task]
model = core/model/classification/roberta
optim = core/optim/adamw
dataset = core/dataset/ast
loss_fn = core/loss/ce
score_fn = core/score/acc
monitor_fns = ['core/score/acc']
output_header = ['query', 'doc', 'label']
post_process_fn = core/postprocess/classfier_score
writer_fn = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 64
dev_batch_size = 64
test_batch_size = 256
```

##### Step 3: Run Training Command

> `--core/task/supervised_task@train_batch_size 128` Override the parameter setting in config file.

```bash
unitorch-train path/to/config.ini --train_file ./train.tsv --dev_file ./dev.tsv --core/task/supervised_task@train_batch_size 128
```

#### Train with your code

```python
from unitorch.models.roberta import RobertaForClassification, RobertaProcessor
from unitorch.optim import Adamw

model = RobertaForClassification(config_path, num_class=4)
processor = RobertaProcessor(vocab_path, merge_path)
optim = Adamw(model.parameters(), lr=1e-5)

## Your Trainer
## ...

```