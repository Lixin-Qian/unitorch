
## Tasks
> SupervisedTask as an example


##### Training Parameters Description
```python
class SupervisedTask(object):
    @add_default_section_for_function("core/task/supervised_task")
    def train(
        self,
        optim=None,
        loss_fn=None,
        score_fn=None,
        monitor_fns=None,
        scheduler=None,
        from_ckpt_dir="./from_ckpt",
        to_ckpt_dir="./to_ckpt",
        train_batch_size=128,
        dev_batch_size=128,
        pin_memory=True,
        num_workers=4,
        save_optimizer=True,
        save_scheduler=True,
        log_freq=100,
        ckpt_freq=10000,
        grad_acc_step=1,
        max_grad_norm=1.0,
        num_training_samples=1000000000,
        epochs=5,
    ):
        pass
```

> * `optim` [Optimizers](/references/optimizers.md)
> * `loss_fn` loss function 
> * `score_fn` score function, used for save checkpoint
> * `monitor_fns` list of score functions, use for logging while model evaluation
> * `scheduler` [Schedulers](/references/schedulers.md)
> * `from_ckpt_dir` checkpoint folder saved by previous training
> * `to_ckpt_dir` folder to save new checkpoint
> * `{train, dev}_batch_size` batch size for train/dev
> * `{pin_memory, num_workers}` dataloader parameters
> * `{save_optimizer, save_scheduler}` whether to save the optimizer/scheduler states
> * `log_freq` logging frequence
> * `ckpt_freq` checkpoint frequence, also the evaluation frequence on dev set
> * `grad_acc_step` gradient accmulation step
> * `max_grad_norm` gradient clip max number
> * `num_training_samples` total number of training data for iterable dataset
> * `epochs` number of training epochs

##### Inference Parameters Description
```python
class SupervisedTask(object):
    @add_default_section_for_function("core/task/supervised_task")
    def infer(
        self,
        output_header: Optional[List] = None,
        test_batch_size: Optional[int] = 128,
        pin_memory: bool = True,
        num_workers: Optional[int] = 4,
        max_size: Optional[int] = 10000,
        from_ckpt_dir: str = "./from_ckpt",
        output_path: str = "./cache/predict.txt",
        postprocess_workers: Optional[int] = 2,
        nrows_per_sample: Optional[int] = None,
        post_process_fn: str = None,
        writer_fn: str = None,
    ):
        pass
```

> * `output_header` save the fields in input data
> * `test_batch_size` batch size for test
> * `{pin_memory, num_workers}` dataloader parameters
> * `max_size` max size for queene
> * `from_ckpt_dir` checkpoint folder saved by previous training
> * `output_path` inference result file
> * `postprocess_workers` number of the postprocess workers
> * `nrows_per_sample` number of the rows that one sample use, mainly used for continue inference
> * `post_process_fn` post process function
> * `writer_fn` writer function

