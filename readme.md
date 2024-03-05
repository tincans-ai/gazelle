# Gazelle - Joint Speech Language Model

This repository contains modeling code for the Gazelle Joint Speech Language Model. Gazelles are fast and if you really squint at 'jsl' while sleep-derived (and forget the difference between the letters `j` and `g`), it makes sense.

![gazelle wearing headphones, cartoon style](logo.webp)

For some more details, read our [blog post](https://tincans.ai/slm). Join us in [Discord](https://discord.gg/qyC5h3FSzU) as well.

This code is almost entirely ripped from [Huggingface's Llava implementation](https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/models/llava/modeling_llava.py). The main changes are just swapping the image encoder for an audio encoder and naively downsampling the audio inputs to a lower frequency. I also added ability to load model weights from pretrained models so that you can train off this code. 

Pretrained v0.1 model: [huggingface](https://huggingface.co/tincans-ai/gazelle-v0.1)

## Disclaimer

We are backproppin' on a budget, the model is definitely not good enough for anything serious, it is mostly fun to make it do anything, the journey is the reward, so on.

Some basic red-teaming and prompt injection in audio shows that the model employs the same safeguards as Llama chat. We have not extensively tested this and make no warranties, it does not represent our views, etc.

## License

This work is heavily derived from Transformers and Llava, both of which are licensed under Apache 2.0. This work is also licensed under Apache 2.0.

The pretrained checkpoints also derive from Llama 2, which is governed by the [Llama 2 license](https://ai.meta.com/llama/license/). You must agree to these terms if you use the pretrained weights!