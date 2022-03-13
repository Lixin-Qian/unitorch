
<h2 align="Center"> <p> ViT Image Classification </p> </h2>

#### Train with command line
##### Step 1: Prepare Customize Dataset
Prepare tsv file as an example. Also support json/hub dataset etc. The tsv should have two columns as dataset settings below. The image column data type is a string of image path.

##### Step 2: Prepare config.ini File
> Take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/classification/vit.ini) as a template.

***Model Setting***

> The options in [core/model/classification/vit] are setting for vit model.


```ini
[core/model/classification/vit]
pretrained_name = vit-base-patch16-224
num_class = 1000
```

***Dataset Setting***

> * The options in [core/dataset/ast/train] are setting for training data. 
> * `names` is the fields of dataset. For tsv files, it should be the header.
> * `preprocess_functions` is the preprocess functions to convert the raw data into tensors as model inputs.
> * These options settings for train/dev/test are independent with each other.


```ini
[core/dataset/ast/train]
data_files = ${core/cli:train_file}
names = ['image', 'category']
preprocess_functions = ['core/process/vit_image_classification(core/process/read_image(image))', 'core/process/label(category)']
```

***Process Setting***

> These setting are used for the pre/post processing classes.

```ini
[core/process/vit]
pretrained_name = vit-base-patch16-224

[core/process/general]
num_class = 1000
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
model = core/model/classification/vit
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
dataset = core/dataset/ast
loss_fn = core/loss/ce
score_fn = core/score/acc
monitor_fns = ['core/score/acc']
output_header = ['image', 'category']
post_process_fn = core/postprocess/classifier_score
writer_fn = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt

num_workers = 16

train_batch_size = 128
dev_batch_size = 128
test_batch_size = 128
```

##### Step 3: Run Training Command

> `--core/task/supervised_task@train_batch_size 128` Override the parameter setting in config file.

```bash
unitorch-train path/to/config.ini --train_file ./train.tsv --dev_file ./dev.tsv --core/task/supervised_task@train_batch_size 128
```

#### Train with your code

```python
from unitorch.models.vit import ViTForImageClassification, ViTProcessor
from unitorch.optim import Adamw

model = ViTForImageClassification(config_path, num_class=1000)
processor = ViTProcessor(vision_config_path)
optim = Adamw(model.parameters(), lr=1e-5)

## Your Trainer
## ...

```