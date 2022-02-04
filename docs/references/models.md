
## Models

> FasterRCNN as an example.

```python
class GeneralizedRCNN(_GeneralizedRCNN):
    def __init__(self, detectron2_config_path):
        pass

class GeneralizedRCNNProcessor(_GeneralizedRCNNProcessor):
    def __init__(
        self,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        pass
```

> * `detectron2_config_path` is the yaml config file in detectron2 library
> * `pixel_mean` is BGR format pixel mean.
> * `pixel_mean` is BGR format pixel std.

