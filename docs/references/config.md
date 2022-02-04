
## Configuration

>  All Parameters in library are configurable.

###### MASSForGeneration
```python
class MASSForGeneration(_MASSForGeneration):
    @add_default_section_for_function("core/model/generation/mass")
    def generate(
        self,
        tokens_ids,
        num_beams=5,
        decoder_start_token_id=101,
        decoder_end_token_id=102,
        num_return_sequences=1,
        min_gen_seq_length=0,
        max_gen_seq_length=48,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        length_penalty=1.0,
        num_beam_groups=1,
        diversity_penalty=0.0,
        diverse_rate=0.0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    ):
        pass
```

###### Config INI Setting
```ini
[core/model/generation/mass]
pretrained_name = mass-base-uncased
no_repeat_ngram_size = 3
max_gen_seq_length = 15
```

###### Inference CLI
```bash
unitorch-infer examples/configs/generation/mass.ini --train_file train.tsv --dev_file dev.tsv \
    --core/model/generation/mass@num_beams 20
```

> * The parameter `num_beams` will be set to 20.
> * The parameter `no_repeat_ngram_size` will be set to 3.
> * The parameter `num_return_sequences` will be set to 1.

!> The piority of the parameter setting is CLI > INI > PY.

