---
base_model: unsloth/llama-3.2-3b-instruct-bnb-4bit
library_name: peft
model_name: stage_0_corrected
tags:
- base_model:adapter:unsloth/llama-3.2-3b-instruct-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
licence: license
pipeline_tag: text-generation
---

# Model Card for stage_0_corrected

This model is a fine-tuned version of [unsloth/llama-3.2-3b-instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3.2-3b-instruct-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 0.26.1
- Transformers: 5.0.0
- Pytorch: 2.10.0a0+b558c986e8.nv25.11
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```