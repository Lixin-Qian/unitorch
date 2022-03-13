<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/scheduler/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.scheduler`




### **Global Variables**
---------------
- **warmup**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/scheduler/warmup.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.scheduler.warmup`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/scheduler/warmup.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CosineWarmupScheduler`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/scheduler/warmup.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CosineWarmupScheduler.__init__`

```python
__init__(
    optimizer:torch.optim.optimizer.Optimizer,
    num_warmup_steps:int,
    num_training_steps:int,
    num_cycles:Optional[float]=0.5,
    last_epoch:Optional[int]=-1
)
```

Cosine Warmup Scheduler 

**Args:**
 
 - <b>`optimizer`</b>:  a torch optimizer 
 - <b>`num_warmup_steps`</b>:  the warmup steps to scheduler 
 - <b>`num_training_steps`</b>:  the training steps to scheduler 





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/scheduler/warmup.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `LinearWarmupScheduler`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/scheduler/warmup.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `LinearWarmupScheduler.__init__`

```python
__init__(
    optimizer:torch.optim.optimizer.Optimizer,
    num_warmup_steps:int,
    num_training_steps:int,
    last_epoch:Optional[int]=-1
)
```

Linear Warmup Scheduler 

**Args:**
 
 - <b>`optimizer`</b>:  a torch optimizer 
 - <b>`num_warmup_steps`</b>:  the warmup steps to scheduler 
 - <b>`num_training_steps`</b>:  the training steps to scheduler 







---

