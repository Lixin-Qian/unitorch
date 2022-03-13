
<h2 align="Center"> <p> VLP Image Caption & Multi-Modal Generation </p> </h2>

#### Train with command line
##### Step 1: Prepare Customize Dataset
Prepare tsv file as an example. Also support json/hub dataset etc. The tsv should have two(for image caption) or three (for multi-modal generation) columns as dataset settings below. The image column data type is a string of image path.

##### Step 2: Prepare config.ini File
> Image Caption take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/caption/vlp.ini) as a template. <br>
> Multi-Modal Generation take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/generation/vlp.ini) as a template.

***Model Setting***

> The options in [core/model/caption/vlp] & [core/model/generation/vlp] are setting for vlp model.


```ini
[core/model/caption/vlp]
pretrained_name = vlp-coco
no_repeat_ngram_size = 0
max_gen_seq_length = 20
```

***Dataset Setting***

> * The options in [core/dataset/ast/train] are setting for training data. 
> * `names` is the fields of dataset. For tsv files, it should be the header.
> * `preprocess_functions` is the preprocess functions to convert the raw data into tensors as model inputs.
> * These options settings for train/dev/test are independent with each other.


```ini
[core/dataset/ast/train]
data_files = ${core/cli:train_file}
names = ['image', 'text', 'caption']
# for image caption setting
preprocess_functions = ['core/process/vlp_caption(image, caption)']
# for multi-modal generation
preprocess_functions = ['core/process/vlp_generation(image, text, caption)']
```

***Process Setting***

> These setting are used for the pre/post processing classes.

```ini
[core/process/vlp]
pretrained_name = vlp-coco
max_seq_length = 24
max_gen_seq_length = 20
```

***Task Setting***
> * `scheduler` scheduler to be used in task.
> * `loss_fn` loss to be used in task.
> * `score_fn` Metric to be used for save checkpoint.
> * `monitor_fns` Metrics for logging.
> * `output_header` Save these fields for inference.
> * `post_process_fn`, `writer_fn` Post process & writer function.
> * `{train, dev, test}_batch_size` Batch size setting for train/eval/inference.

```ini
[core/task/supervised_task]
model = core/model/caption/vlp # core/model/generation/vlp
optim = core/optim/adamw
dataset = core/dataset/ast
loss_fn = core/loss/lm
score_fn = core/score/bleu
monitor_fns = ['core/score/bleu', 'core/score/rouge1', 'core/score/rouge2', 'core/score/rougel']
output_header = ['image', 'text', 'caption']
post_process_fn = core/postprocess/vlp_detokenize
writer_fn = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 8
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
from unitorch.models.vlp import VLPForGeneration, VLPProcessor
from unitorch.optim import Adamw

model = VLPForGeneration(vlp_config_path, detectron2_config_path)
processor = VLPProcessor(vocab_path, pixel_mean=[103.53, 116.28, 123.675], pixel_std=[1.0, 1.0, 1.0])
optim = Adamw(model.parameters(), lr=1e-5)

## Your Trainer
## ...

```