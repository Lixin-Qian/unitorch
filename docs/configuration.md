
## Configuration in unitorch adapter classes

> Take BartForGeneration class as an example

#### source code

```python
class BartForGeneration(_BartForGeneration):
    def __init__(
        self,
        config_path,
        gradient_checkpointing=False,
    ):
        pass

    @classmethod
    @add_default_section_for_init("core/model/generation/bart")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/generation/bart")
        pretrained_name = config.getoption("pretrained_name", "default-bart")
        config_name_or_path = config.getoption("config_path", pretrained_name)
        config_path = (
            pretrained_bart_infos[config_name_or_path]["config"]
            if config_name_or_path in pretrained_bart_infos
            else config_name_or_path
        )
        return {"config_path": config_path}

    @add_default_section_for_function("core/model/generation/bart")
    def generate(
        self,
        tokens_ids,
        num_beams=5,
        decoder_start_token_id=2,
        decoder_end_token_id=2,
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

#### configuration

```ini
[core/model/generation/bart]
pretrained_name = bart-base
no_repeat_ngram_size = 3
max_gen_seq_length = 15
```

#### unitorch-cli

```bash
unitorch-train \ 
    configs/core/generation/bart.ini \
    --train_file train.tsv \
    --dev_file dev.tsv \
    --core/model/generation/bart@num_beams 20 \
    --core/model/generation/bart@no_repeat_ngram_size 0
```

!> unitorch cli > configuration > default python parameter value


#### **Final Parameter Setting**
> * `config_path`: use the bart-base pretrained config.json
> * `no_repeat_ngram_size`: 0
> * `max_gen_seq_length`: 15
> * `num_beams`: 0
> * `num_return_sequences`: 1

