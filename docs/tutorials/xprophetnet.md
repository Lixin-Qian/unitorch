
<h2 align="Center"> <p> XProphetNet Multi-Lingual Generation </p> </h2>

#### Train with command line
##### Step 1: Prepare Customize Dataset
Prepare tsv file as an example. Also support json/hub dataset etc. The tsv should have two columns as dataset settings below.

##### Step 2: Prepare config.ini File
> Take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/generation/xprophetnet.ini) as a template.

***Model Setting***
> The options in [core/model/generation/xprophetnet] are setting for xprophetnet model.
 

```ini
[core/model/generation/xprophetnet]
pretrained_name = xprophetnet-large-wiki100-cased
num_beams = 20
```

***Dataset Setting***

> * The options in [core/dataset/ast/train] are setting for training data. 
> * `names` is the fields of dataset. For tsv files, it should be the header.
> * `preprocess_functions` is the preprocess functions to convert the raw data into tensors as model inputs.
> * These options settings for train/dev/test are independent with each other.


```ini
[core/dataset/ast/train]
data_files = ${core/cli:train_file}
names = ['encode', 'decode']
preprocess_functions = ['core/process/xprophetnet_generation(encode, decode)']

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
names = ['encode', 'decode']
preprocess_functions = ['core/process/xprophetnet_inference(encode)', 'core/process/xprophetnet_evaluation(decode)']
```

***Process Setting***

> These setting are used for the pre/post processing classes.

```ini
[core/process/xprophetnet]
pretrained_name = xprophetnet-large-wiki100-cased
max_seq_length = 24
max_gen_seq_length = 15
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
model = core/model/generation/xprophetnet
optim = core/optim/adamw
dataset = core/dataset/ast
loss_fn = core/loss/prophetnet
score_fn = core/score/bleu
monitor_fns = ['core/score/bleu', 'core/score/rouge1', 'core/score/rouge2', 'core/score/rougel']
output_header = ['encode', 'decode']
post_process_fn = core/postprocess/xprophetnet_detokenize
writer_fn = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 4
dev_batch_size = 8
test_batch_size = 8
```

##### Step 3: Run Training Command

> `--core/task/supervised_task@train_batch_size 8` Override the parameter setting in config file.

```bash
unitorch-train path/to/config.ini --train_file ./train.tsv --dev_file ./dev.tsv --core/task/supervised_task@train_batch_size 8
```

#### Train with your code

```python
from unitorch.models.xprophetnet import XProphetNetForGeneration, XProphetNetProcessor
from unitorch.optim import Adamw

model = XProphetNetForGeneration(config_path)
processor = XProphetNetProcessor(vocab_path)
optim = Adamw(model.parameters(), lr=1e-5)

## Your Trainer
## ...

```