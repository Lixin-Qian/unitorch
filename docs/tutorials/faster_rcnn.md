
<h2 align="Center"> <p> FasterRCNN Object Detection </p> </h2>

#### Train with command line
##### Step 1: Prepare Customize Dataset
Prepare tsv file as an example. Also support json/hub dataset etc. The tsv should have three columns as dataset settings below. The image column data type is a string of image path.

##### Step 2: Prepare config.ini File
> Take [this config](https://github.com/fuliucansheng/unitorch/examples/configs/core/detection/faster_rcnn.ini) as a template.

***Model Setting***
> The options in [core/model/detection/generalized_rcnn] are setting for roberta model.
 

```ini
[core/model/detection/generalized_rcnn]
pretrained_name = pascal-voc-detection/faster-rcnn-r50-c4
```

***Dataset Setting***

> * The options in [core/dataset/ast/train] are setting for training data. 
> * `names` is the fields of dataset. For tsv files, it should be the header.
> * `preprocess_functions` is the preprocess functions to convert the raw data into tensors as model inputs.
> * These options settings for train/dev/test are independent with each other.


```ini
[core/dataset/ast/train]
data_files = ${core/cli:train_file}
names = ['image', 'bboxes', 'classes']
preprocess_functions = ['core/process/generalized_rcnn_detection(core/process/read_image(image), bboxes, classes)']

[core/dataset/ast/dev]
data_files = ${core/cli:dev}
names = ['image', 'bboxes', 'classes']
preprocess_functions = ['core/process/generalized_rcnn_detection_evaluation(core/process/read_image(image), bboxes, classes)']
```

***Process Setting***

> These setting are used for the pre/post processing classes.

```ini
[core/process/generalized_rcnn]
pixel_mean = [103.53, 116.28, 123.675]
pixel_std = [1.0, 1.0, 1.0]
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
model = core/model/detection/generalized_rcnn
optim = core/optim/adamw
dataset = core/dataset/ast
score_fn = core/score/voc_map
monitor_fns = ['core/score/voc_map']
output_header = ['image', 'bboxes', 'classes']
post_process_fn = core/postprocess/generalized_rcnn
writer_fn = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 4
dev_batch_size = 1
test_batch_size = 1
```

##### Step 3: Run Training Command

> `--core/task/supervised_task@train_batch_size 8` Override the parameter setting in config file.

```bash
unitorch-train path/to/config.ini --train_file ./train.tsv --dev_file ./dev.tsv --core/task/supervised_task@train_batch_size 8
```

#### Train with your code

```python
from unitorch.models.vit import GeneralizedRCNN, ViTProcessor
from unitorch.optim import Adamw

model = GeneralizedRCNN(config_path)
processor = GeneralizedRCNNProcessor(pixel_mean=[103.53, 116.28, 123.675], pixel_std=[1.0, 1.0, 1.0])
optim = Adamw(model.parameters(), lr=1e-5)

## Your Trainer
## ...

```