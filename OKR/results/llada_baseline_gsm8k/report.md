# LLaDA Baseline GSM8K Report

## Setup
- model_name_or_path: models/LLaDA-8B-Instruct
- dataset: gsm8k (datasets/gsm8k)
- split: test
- seed: 112
- prompt_template: collate_fn_gsm8k (plain question)
- gen_params: steps=256, gen_length=256, block_length=8, no_sample=true
- distributed: nproc_per_node=2 (torchrun)

## Metrics
- accuracy: 78.79% (0.7879)
- num_examples: 1319

