<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/utils/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.utils`




### **Global Variables**
---------------
- **buffer**

- **decorators**

- **palette**





---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/utils/buffer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.utils.buffer`








---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/utils/decorators.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.utils.decorators`




### **Global Variables**
---------------
- **OPTIMIZED_CLASSES**

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/utils/decorators.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `replace`

```python
replace(target_obj)
```

A decorator to replace the specified obj. 

`target_obj` can be a class or a function. 



**Example:**
 

```python
class A:
     def f(self):
         print('class A')
@replace(A)
class B:
     def f(self):
         print('class B')
``` 



**Args:**
 
 - <b>`target_obj`</b> (class/func/method):  a class, method, or function to be  replaced. 



**Returns:**
 A decorator function to replace the input object. 




---


<!-- markdownlint-disable -->

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/utils/palette.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>module</kbd> `unitorch.utils.palette`




### **Global Variables**
---------------
- **default**
- **pascal**
- **palette**

---

<a href="https://github.com/fuliucansheng/unitorch/blob/master/unitorch/utils/palette.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>function</kbd> `get`

```python
get(name)
```








---

