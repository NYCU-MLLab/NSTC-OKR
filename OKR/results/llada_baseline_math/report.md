# LLaDA Baseline MATH Report

## Setup
- model_name_or_path: models/LLaDA-8B-Instruct
- dataset: MATH (datasets/MATH)
- split: test
- seed: 112
- prompt_template: collate_fn_math
- gen_params: steps=256, gen_length=256, block_length=8, no_sample=true
- distributed: nproc_per_node=2 (torchrun)

## Metrics
- accuracy: 39.24% (0.3924)
- num_examples: 5000
