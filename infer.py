#!/usr/bin/env python
# coding: utf-8

# Inference example
# This notebook requires 24GB of vRAM. The underlying models should be able to be quantized lower but it's pretty confusing how to do so within transformers.

import argparse
import string
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
    "--data-type", "-d", help="Data type to use for the model", default="bfloat16"
)
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

if torch.cuda.is_available():
    model = model.cuda()


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
        "audio_values": audio_values.squeeze(0).to(model.device).to(torch.bfloat16),
        "input_ids": labels.to(model.device),
    }


def infer(audio_file: str, prompt: str):
    test_audio, sr = torchaudio.load(audio_file)
    if sr != 16000:
        test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
    inputs = inference_collator(test_audio, prompt)
    tokens = input_ids = inputs["input_ids"]
    temperature = args.temperature if args.temperature > 0.0 else None
    do_sample = temperature is not None
    prev_text = ""
    while tokens.shape[1] - input_ids.shape[1] < args.max_tokens:
        attention_mask = torch.ones(tokens.shape[1], device=tokens.device).unsqueeze(0)
        tokens = model.generate(
            tokens,
            do_sample=do_sample,
            max_length=tokens.shape[1] + 1,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            attention_mask=attention_mask,
        )  # [1 x M + 1]
        new_token = tokens[0][-1]
        if new_token.item() == tokenizer.eos_token_id:
            break

        # Decode all output tokens; this ensure proper space insertion.
        text = tokenizer.decode(tokens[0][input_ids.shape[1] :])
        yield text[len(prev_text) :]
        prev_text = text


if __name__ == "__main__":
    for token in infer(args.audio_file, args.prompt):
        print(token, end="", flush=True)
    print("\n")
