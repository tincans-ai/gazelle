# Gazelle - Joint Speech Language Model

This repository contains modeling code for the Gazelle Joint Speech Language Model. Gazelles are fast and if you really squint at 'jsl' while sleep-derived (and forget the difference between the letters `j` and `g`), it makes sense.

![gazelle wearing headphones, cartoon style](logo.webp)

For some more details, read our [blog post](https://tincans.ai/slm).

This code is almost entirely ripped from [Huggingface's Llava implementation](https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/models/llava/modeling_llava.py). The main changes are just swapping the image encoder for an audio encoder and naively downsampling the audio inputs to a lower frequency. I also added ability to load model weights from pretrained models so that you can train off this code.

Pretrained weights and instructions to follow soon. 

## Disclaimer

The initial release of the model is not good! It was not trained enough and used a dataset that I had lying around that is not what you want to use! Consider it a proof of concept only! 

## Notes / advice

Training notes:

- torch compile makes a large difference. There are currently still graph breaks that I don't have the time to debug.
- I forced everything to bfloat16. I suspect that precision loss is not that important since the big models are frozen.
- I used cuda 12.1 mostly. I ran into lots of problems in 11.8 with random inconsistencies.
- FSDP and DeepSpeed should _theoretically_ work but my original dataloader was not happy with multi-GPU setup.
- I used the Huggingface trainer. It should be reasonably straightforward to set it up.

Dependency notes:

- I have not tested on older versions of Transformers or PyTorch but they should work. I've found Torch 2.2 to be significantly faster in training because of improvements to `torch.compile` and `flash-attn`. 
- SentencePiece is needed for Llama.
- I used `adamw_8bit_bnb` in training because it seems like free lunch. Needs `accelerate` and `bitsandbytes` for training.
- Make sure that you're logged into HF or `HF_TOKEN` is set, to load the Llama 2 checkpoint from HF.

Hyperparameter notes:

- I did not conduct any sweeps. Most numbers were borrowed from Llava's 1.5 runs.
- Stacking factor of 8 is arbitrary and possibly too low. FB papers suggested 24 as an efficient number that maintained good quality while cutting the audio inputs down a lot. You want a factor of 8 because of CUDA shenanigans though.
- A 'real' ML researcher insists that the 2e-3 learning rate for pretrain is way too low, based on the weird loss curve. [Ma et al 2023](https://arxiv.org/abs/2402.08846) saw a very similar loss curve training for ASR, maybe they also had too low of LR.
- I used batch size 64, lower numbers seemed bad. Did not test higher numbers, it would have had to be gradient accumulation anyways. I suspect higher batch size helps.


## License

This work is heavily derived from Transformers and Llava, both of which are licensed under Apache 2.0. This work is also licensed under Apache 2.0.

The pretrained checkpoints also derive from Llama 2, which is governed by the [Llama 2 license](https://ai.meta.com/llama/license/). You must agree to these terms if you use the pretrained weights!