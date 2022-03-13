<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.loss`




### **Global Variables**
---------------
- **prophetnet**



---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `CELoss`
Creates a criterion that measures the Cross Entropy between the target and the input probabilities. 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CELoss.__init__`

```python
__init__(
    smoothing_alpha:Optional[float]=0.0,
    weight:Optional[torch.Tensor]=None,
    reduction:Optional[str]='mean'
)
```



**Args:**
 
 - <b>`smoothing_alpha`</b> (float):  alpha to smoothing label. 
 - <b>`weight`</b> (torch.Tensor, optional):  a manual rescaling weight given to the loss of each batch element. 
 - <b>`reduction`</b> (string, optional):  specifies the reduction to apply to the output. 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `CELoss.forward`

```python
forward(
    input:torch.Tensor,
    target:torch.Tensor,
    sample_weight:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`input`</b>:  output tensor from model 
 - <b>`target`</b>:  target tensor for model 
 - <b>`sample_weight`</b>:  weight for each sample in a batch 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `BCELoss`
Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities. 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BCELoss.__init__`

```python
__init__(weight:Optional[torch.Tensor]=None, reduction:str='mean')
```



**Args:**
 
 - <b>`weight`</b> (torch.Tensor, optional):  a manual rescaling weight given to the loss of each batch element. 
 - <b>`reduction`</b> (string, optional):  specifies the reduction to apply to the output. 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `BCELoss.forward`

```python
forward(
    input:torch.Tensor,
    target:torch.Tensor,
    sample_weight:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`input`</b>:  output tensor from model 
 - <b>`target`</b>:  target tensor for model 
 - <b>`sample_weight`</b>:  weight for each sample in a batch 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `LMLoss`
Creates a criterion used for language model 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `LMLoss.__init__`

```python
__init__(reduction:str='mean')
```



**Args:**
 
 - <b>`reduction`</b> (string, optional):  specifies the reduction to apply to the output. 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `LMLoss.forward`

```python
forward(
    input:torch.Tensor,
    target:torch.Tensor,
    masks:Optional[torch.Tensor]=None,
    sample_weight:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`input`</b>:  output tensor from model 
 - <b>`target`</b>:  target tensor for model 
 - <b>`masks`</b>:  mask matrix for target 
 - <b>`sample_weight`</b>:  weight for each sample in a batch 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `MSELoss`
Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MSELoss.__init__`

```python
__init__(reduction:str='mean')
```



**Args:**
 
 - <b>`reduction`</b> (string, optional):  specifies the reduction to apply to the output. 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/__init__.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `MSELoss.forward`

```python
forward(
    input:torch.Tensor,
    target:torch.Tensor,
    sample_weight:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`input`</b>:  output tensor from model 
 - <b>`target`</b>:  target tensor for model 
 - <b>`sample_weight`</b>:  weight for each sample in a batch 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/prophetnet.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.loss.prophetnet`






---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/prophetnet.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>class</kbd> `ProphetnetLoss`
Creates a criterion for prophetnet 

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/prophetnet.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `ProphetnetLoss.__init__`

```python
__init__(reduction:str='mean')
```



**Args:**
 
 - <b>`reduction`</b> (string, optional):  specifies the reduction to apply to the output. 




---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/loss/prophetnet.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

##### <kbd>method</kbd> `ProphetnetLoss.forward`

```python
forward(
    input:torch.Tensor,
    target:torch.Tensor,
    masks:Optional[torch.Tensor]=None,
    sample_weight:Optional[torch.Tensor]=None
)
```



**Args:**
 
 - <b>`input`</b>:  output tensor from model 
 - <b>`target`</b>:  target tensor for model 
 - <b>`masks`</b>:  mask matrix for target 
 - <b>`sample_weight`</b>:  weight for each sample in a batch 




---

