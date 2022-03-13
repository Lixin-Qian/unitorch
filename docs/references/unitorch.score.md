<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.score`




### **Global Variables**
---------------
- **bleu**

- **rouge**

- **voc_map**




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/bleu.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.score.bleu`





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/bleu.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `bleu_score`

```python
bleu_score(
    y_true:List[Union[str, int, List[Union[str, int]]]],
    y_pred:List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens:Optional[List[Union[str, int]]]=None
)
```



**Args:**
 
 - <b>`y_true`</b>:  list of lists of int/str tokens of ground truth. 
 - <b>`y_pred`</b>:  list of lists of int/str tokens of generation results. 
 - <b>`ignore_tokens`</b>:  the token list to filtration 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/rouge.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.score.rouge`





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/rouge.py#L287"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `rouge1_score`

```python
rouge1_score(
    y_true:List[Union[str, int, List[Union[str, int]]]],
    y_pred:List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens:Optional[List[Union[str, int]]]=None
)
```



**Args:**
 
 - <b>`y_true`</b>:  list of lists of int/str tokens of ground truth. 
 - <b>`y_pred`</b>:  list of lists of int/str tokens of generation results. 
 - <b>`ignore_tokens`</b>:  the token list to filtration 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/rouge.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `rouge2_score`

```python
rouge2_score(
    y_true:List[Union[str, int, List[Union[str, int]]]],
    y_pred:List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens:Optional[List[Union[str, int]]]=None
)
```



**Args:**
 
 - <b>`y_true`</b>:  list of lists of int/str tokens of ground truth. 
 - <b>`y_pred`</b>:  list of lists of int/str tokens of generation results. 
 - <b>`ignore_tokens`</b>:  the token list to filtration 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/rouge.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `rougel_score`

```python
rougel_score(
    y_true:List[Union[str, int, List[Union[str, int]]]],
    y_pred:List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens:Optional[List[Union[str, int]]]=None
)
```



**Args:**
 
 - <b>`y_true`</b>:  list of lists of int/str tokens of ground truth. 
 - <b>`y_pred`</b>:  list of lists of int/str tokens of generation results. 
 - <b>`ignore_tokens`</b>:  the token list to filtration 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/voc_map.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.score.voc_map`





---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/voc_map.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `voc_ap_score`

```python
voc_ap_score(
    p_bboxes:List[numpy.ndarray],
    p_scores:List[numpy.ndarray],
    p_classes:List[numpy.ndarray],
    gt_bboxes:List[numpy.ndarray],
    gt_classes:List[numpy.ndarray],
    class_id:int=None,
    threshold:float=0.5
)
```



**Args:**
 
 - <b>`p_bboxes`</b>:  a list of predict bboxes 
 - <b>`p_scores`</b>:  a list of predict score for bbox 
 - <b>`p_classes`</b>:  a list of predict class id for bbox 
 - <b>`gt_bboxes`</b>:  a list of ground truth bboxes 
 - <b>`gt_classes`</b>:  a list of true class id for each true bbox 
 - <b>`class_id`</b>:  the class id to compute ap score 
 - <b>`threshold`</b>:  the threshold to ap score 


---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/score/voc_map.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `voc_map_score`

```python
voc_map_score(
    p_bboxes:List[numpy.ndarray],
    p_scores:List[numpy.ndarray],
    p_classes:List[numpy.ndarray],
    gt_bboxes:List[numpy.ndarray],
    gt_classes:List[numpy.ndarray]
)
```



**Args:**
 
 - <b>`p_bboxes`</b>:  a list of predict bboxes 
 - <b>`p_scores`</b>:  a list of predict score for bbox 
 - <b>`p_classes`</b>:  a list of predict class id for bbox 
 - <b>`gt_bboxes`</b>:  a list of ground truth bboxes 
 - <b>`gt_classes`</b>:  a list of true class id for each true bbox 

**Returns:**
 a avg ap score of all classes in ground truth 




---

