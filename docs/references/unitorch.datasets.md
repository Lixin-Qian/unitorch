<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.datasets`




### **Global Variables**
---------------
- **huggingface**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.datasets.huggingface`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `hf_datasets`
A dataclass of huggingface datasets library https://github.com/huggingface/datasets 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `hf_datasets.__init__`

```python
__init__(dataset:datasets.arrow_dataset.Dataset)
```

A class based on huggingface datasets `dataset` is an instance of huggingface dataset 


---

###### <kbd>property</kbd> hf_datasets.dataset

The property of actual hf dataset 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_datasets.from_csv`

```python
from_csv(
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    names:Optional[List[str]]=None,
    sep:Optional[str]='\t',
    split:Optional[str]=None
)
```

A classmethod of load csv/tsv/text files dataset 

**Args:**
 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`names`</b>:  header names to data file(s). 
 - <b>`sep`</b>:  seperator for text file(s). 
 - <b>`split`</b>:  which split of the data to load. Returns: return a dataset. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_datasets.from_hub`

```python
from_hub(
    data_name,
    config_name=None,
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    split:Optional[str]=None
)
```

A classmethod of load hf hub dataset 

**Args:**
 
 - <b>`data_name`</b>:  a dataset repository on the hf hub. 
 - <b>`config_name`</b>:  defining the name of the dataset configuration. 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`split`</b>:  which split of the data to load. Returns: return a dataset. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_datasets.from_json`

```python
from_json(
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    field:Optional[str]=None,
    split:Optional[str]=None
)
```

A classmethod of load json files dataset 

**Args:**
 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`field`</b>:  specify the field to load in json file. 
 - <b>`split`</b>:  which split of the data to load. Returns: return a dataset. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_datasets.from_parquet`

```python
from_parquet(
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    split:Optional[str]=None
)
```

A classmethod of load parquet files dataset 

**Args:**
 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`split`</b>:  which split of the data to load. Returns: return a dataset. 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `hf_iterable_datasets`
A dataclass of huggingface datasets library https://github.com/huggingface/datasets 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `hf_iterable_datasets.__init__`

```python
__init__(dataset:datasets.arrow_dataset.Dataset)
```






---

###### <kbd>property</kbd> hf_iterable_datasets.dataset

The property of actual hf dataset 



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_iterable_datasets.from_csv`

```python
from_csv(
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    names:Optional[List[str]]=None,
    sep:Optional[str]='\t',
    split:Optional[str]=None
)
```

A classmethod of load csv/tsv/text files dataset 

**Args:**
 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`names`</b>:  header names to data file(s). 
 - <b>`sep`</b>:  seperator for text file(s). 
 - <b>`split`</b>:  which split of the data to load. Returns: return a streaming dataset. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_iterable_datasets.from_hub`

```python
from_hub(
    data_name,
    config_name:Optional[str]=None,
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    split:Optional[str]=None
)
```

A classmethod of load hf hub dataset 

**Args:**
 
 - <b>`data_name`</b>:  a dataset repository on the hf hub. 
 - <b>`config_name`</b>:  defining the name of the dataset configuration. 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`split`</b>:  which split of the data to load. Returns: return a streaming dataset 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_iterable_datasets.from_json`

```python
from_json(
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    field:Optional[str]=None,
    split:Optional[str]=None
)
```

A classmethod of load json files dataset 

**Args:**
 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`field`</b>:  specify the field to load in json file. 
 - <b>`split`</b>:  which split of the data to load. Returns: return a streaming dataset. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>classmethod</kbd> `hf_iterable_datasets.from_parquet`

```python
from_parquet(
    data_dir:Optional[str]=None,
    data_files:Optional[str, List[str]]=None,
    split:Optional[str]=None
)
```

A classmethod of load parquet files dataset 

**Args:**
 
 - <b>`data_dir`</b>:  defining the data_dir of the dataset configuration. 
 - <b>`data_files`</b>:  path(s) to source data file(s). 
 - <b>`split`</b>:  which split of the data to load. Returns: return a streaming dataset. 

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/datasets/huggingface.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `hf_iterable_datasets.set_epoch`

```python
set_epoch(epoch)
```








---

