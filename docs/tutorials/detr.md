
<h2 align="Center"> <p> DETR Segmentation </p> </h2>

#### Train with command line
##### Step 1: Prepare Customize Dataset
Prepare tsv file as an example. Also support json/hub dataset etc. The tsv should have four columns as dataset settings below.

##### Step 2: Prepare config.ini File
> Take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/segmentation/detr.ini) as a template.

***Model Setting***
> The options in [core/model/segmentation/detr] are setting for detr model.
 

```ini
[core/model/segmentation/detr]
pretrained_name = detr-resnet-50
```

***Dataset Setting***

> * The options in [core/dataset/ast/train] are setting for training data. 
> * `names` is the fields of dataset. For tsv files, it should be the header.
> * `preprocess_functions` is the preprocess functions to convert the raw data into tensors as model inputs.
> * These options settings for train/dev/test are independent with each other.


```ini
[core/dataset/ast/train]
data_files = ${core/cli:train_file}
names = ['image', 'bboxes', 'classes', 'gt_image']
preprocess_functions = ['core/process/detr_segmentation(core/process/read_image(image), core/process/read_image(gt_image), bboxes, classes)']
```

***Process Setting***

> These setting are used for the pre/post processing classes.

```ini
[core/process/detr]
pretrained_name = detr-resnet-50
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
model = core/model/segmentation/detr
optim = core/optim/adamw
dataset = core/dataset/ast
score_fn = core/score/acc
monitor_fns = ['core/score/acc']
output_header = ['image', 'bboxes', 'classes', 'gt_image']
post_process_fn = core/postprocess/detr_segmentation
writer_fn = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 1
dev_batch_size = 1
test_batch_size = 1

```

##### Step 3: Run Training Command

> `--core/task/supervised_task@train_batch_size 128` Override the parameter setting in config file.

```bash
unitorch-train path/to/config.ini --train_file ./train.tsv --dev_file ./dev.tsv --core/task/supervised_task@train_batch_size 128
```

#### Train with your code

```python
from unitorch.models.detr import DetrForSegmentation, DetrProcessor
from unitorch.optim import Adamw

model = DetrForSegmentation(config_path)
processor = DetrProcessor(vision_config_path)
optim = Adamw(model.parameters(), lr=1e-5)

## Your Trainer
## ...

```