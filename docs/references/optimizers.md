
## Optimizers

> AdamW as an example.

```python
class AdamWOptimizer(AdamW, OptimMixin):
    def __init__(
        self,
        params,
        learning_rate: float = 0.00001,
    ):
        pass
```

> * `learning_rate` lr for optimizer