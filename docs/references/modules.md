
## Modules

> BeamSearchScorerV2 as an example.

```python
@replace(BeamSearchScorer)
class BeamSearchScorerV2(BeamSearchScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        use_reorder_cache_v2: Optional[bool] = False,
    ):
        pass
```

> * `BeamSearchScorer` is from huggingface transformers.
> * `BeamSearchScorerV2` is to optimize the offical one for faster inference by supporting `reorder_cache_v2` function in model classes.
> * `replace` decorator is a useful function that can replace all the `BeamSearchScorer` with `BeamSearchScorerV2` in all the imported modules.

!> This is a better solution if we need to optimize some code in the library we have imported.