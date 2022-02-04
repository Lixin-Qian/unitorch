
## Quick Start

#### Installation
```bash
pip install unitorch
```

#### Command Line

###### *With Dataset From Huggingface Hub*
> The [Hub Dataset](https://huggingface.co/datasets/fuliucansheng/mininlp) is configed in the ini file.
```bash
unitorch-train examples/configs/core/classification/roberta.ini
```

###### *With Local Files*
> For default generation config files, the data need to be a tsv file with [encode, decode] format.
```bash
unitorch-train examples/configs/core/generation/bart.ini --train_file ./train.tsv --dev_file ./dev.tsv
```

#### Python Package

Simply add one-line code `import unitorch` if using this package.

###### *Use Unilm Model Class*

```python
from unitorch.models.unilm import UnilmForGeneration
unilm_model = UnilmForGeneration("path/to/unilm/config.json")
```

##### *Use Configuration Class*

```python
from unitorch import CoreConfigureParser
config = CoreConfigureParser("path/to/config.ini")
```