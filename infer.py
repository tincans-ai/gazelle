#!/usr/bin/env python
# coding: utf-8

# Inference example
# This notebook requires 24GB of vRAM. The underlying models should be able to be quantized lower but it's pretty confusing how to do so within transformers.

import argparse
import time
import torch
import torchaudio
import transformers

from gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
)

parser = argparse.ArgumentParser()
parser.add_argument("audio_file", help="Path to the audio file")
parser.add_argument(
    "prompt", nargs="?", default="Listen to <|audio|> and respond to it"
)
parser.add_argument(
    "--model",
    "-m",
    help="Model ID to use for the model",
    default="tincans-ai/gazelle-v0.2",
)
parser.add_argument(
    "--device",
    "-D",
    help="Device to use for inference",
)
parser.add_argument("--data-type", help="Data type to use for the model")
parser.add_argument(
    "--temperature", "-t", help="Temperature for sampling", default=0.0, type=float
)
parser.add_argument(
    "--repetition-penalty", "-r", help="Repetition penalty", default=1.0, type=float
)
parser.add_argument(
    "--max-tokens", "-T", help="Maximum tokens to generate", default=64, type=int
)
args = parser.parse_args()

device = args.device or (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
data_type = args.data_type or ("bfloat16" if torch.cuda.is_available() else "float32")
dtype = (
    torch.bfloat16
    if args.data_type == "bfloat16"
    else torch.float16 if args.data_type == "float16" else torch.float32
)


model_id = args.model
config = GazelleConfig.from_pretrained(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = GazelleForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model = model.to(device).to(dtype)


def inference_collator(audio_input, prompt: str):
    audio_values = audio_processor(
        audio=audio_input, return_tensors="pt", sampling_rate=16000
    ).input_values
    msgs = [
        {"role": "user", "content": prompt},
    ]
    labels = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    return {
        "audio_values": audio_values.squeeze(0).to(model.device).to(dtype),
        "input_ids": labels.to(model.device),
    }


def infer(audio_file: str, prompt: str):
    test_audio, sr = torchaudio.load(audio_file)
    if sr != 16000:
        test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
    inputs = inference_collator(test_audio, prompt)
    input_len = inputs["input_ids"].shape[1]
    temperature = args.temperature or None
    repetition_penalty = args.repetition_penalty or None
    do_sample = temperature is not None

    output = model.generate(
        **inputs,
        do_sample=do_sample,
        max_length=input_len + args.max_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    output_len = output.shape[1] - input_len
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True), output_len


if __name__ == "__main__":
    start_time = time.time()
    result, num_tokens = infer(args.audio_file, args.prompt)
    elapsed = time.time() - start_time
    print(result)
    print(
        f"Generated {num_tokens} tokens in {elapsed:.2f} seconds, {num_tokens / elapsed:.2f} tps"
    )
