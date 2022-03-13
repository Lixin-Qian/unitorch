<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.modules`




### **Global Variables**
---------------
- **replace**

- **classifier**

- **prefix_model**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.modules.classifier`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `reslayer`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `reslayer.__init__`

```python
__init__(
    feature_dim:int,
    downscale_dim:int,
    output_dim:int,
    use_bn:Optional[bool]=True
)
```

Net Structure: ` in -> fc1 -> bn -> relu -> fc2 + in -> relu` 

**Args:**
 
    - feature_dim: the input feature dim 
    - downscale_dim: the downscale dim 
    - output_dim: the output dim 
    - use_bn: if add bn between fc1 & relu 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `reslayer.forward`

```python
forward(input:torch.Tensor)
```



**Args:**
 
 - <b>`input`</b>:  the input 2d tensor 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `mlplayer`




<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `mlplayer.__init__`

```python
__init__(
    feature_dim:int,
    downscale_dim:int,
    output_dim:int,
    add_pre_layer_norm:Optional[bool]=True,
    add_post_layer_norm:Optional[bool]=False
)
```

Net Structure: ` in -> pre_layer_norm -> fc1 -> gelu -> fc2 + in -> post_layer_norm` 

**Args:**
 
    - feature_dim: the input feature dim 
    - downscale_dim: the downscale dim 
    - output_dim: the output dim 
    - add_pre_layer_norm: if use pre_layer_norm between in & fc1 
    - add_post_layer_norm: if use post_layer_norm after fc2 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/modules/classifier.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `mlplayer.forward`

```python
forward(input:torch.Tensor)
```



**Args:**
 
 - <b>`input`</b>:  the input 2d tensor 




---

