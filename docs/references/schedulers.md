
## Schedulers

> Linear Warmup Scheduler as an example

```python
class LinearWarmupScheduler(LinearWarmupScheduler, SchedulerMixin):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        pass
```

> * `num_warmup_steps` number of warmup steps
> * `num_training_steps` number of total training steps

