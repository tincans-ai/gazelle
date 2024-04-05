import argparse
import io
import json
import math
import os
from dataclasses import dataclass

import librosa
import numpy as np
import requests
import torch
from bitsandbytes.optim import AdamW8bit
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForSeq2Seq, LlamaTokenizerFast, Wav2Vec2Processor

from gazelle import GazelleConfig, GazelleForConditionalGeneration, GazelleProcessor

TRANSCRIBE_PROMPT = (
    "Transcribe the spoken words from <|audio|> with exact wording and punctuation"
)
ANSWER_PROMPT = "Listen to <|audio|> and respond to it"
TRANSCRIBE_INPUT_TASK = "transcribe_input"
TRANSCRIBE_OUTPUT_TASK = "transcribe_output"
ANSWER_TASK = "answer"

parser = argparse.ArgumentParser()
parser.add_argument("--metadata-file", "-m", type=str)
parser.add_argument("--audio-dir", "-a", type=str)
parser.add_argument("--batch-size", "-b", type=int, default=4)
parser.add_argument("--data-type", "-d", type=str, default="bfloat16")
parser.add_argument("--optimizer", "-o", type=str, default="adamw_bnb_8bit")
parser.add_argument("--num-samples", "-n", type=int)
parser.add_argument("--num-epochs", "-e", type=int, default=1)
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()
dtype = (
    torch.bfloat16
    if args.data_type == "bfloat16"
    else torch.float16 if args.data_type == "float16" else torch.float32
)


@dataclass
class DataCollatorForSeq2SeqWithAudio(DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        audio_features = [f.pop("audio_values") for f in features]
        batch = super().__call__(features, *args, **kwargs)
        batch["audio_values"] = torch.nn.utils.rnn.pad_sequence(
            audio_features, batch_first=True
        )

        return batch


class AnyInstructSpeechDataset(Dataset):
    """
    Metadata file format:
    {"chat": [
        {"role": "USER", "message": "Write a sentence based on this summary: iraqi embassy in jakarta removes saddam hussein 's photo", "speech": "chunk_00000/0001.mp3"},
        {"role": "AnyGPT", "message": "The building in Jakarta where people from Iraq work, took down a picture of a man named Saddam Hussein.", "speech": "chunk_00000/0002.mp3"}
    ]}
    """

    def __init__(self, args, processor):
        self.session = requests.Session()
        self.audio_dir = args.audio_dir
        self.processor = processor
        if args.metadata_file:
            print(f"Loading metadata from {args.metadata_file}...")
            with open(args.metadata_file, "r") as f:
                jsonl = [json.loads(line) for line in f.readlines()]
        else:
            print("Loading metadata from Hugging Face dataset...")
            response = self.session.get(
                "https://huggingface.co/datasets/fnlp/AnyInstruct/resolve/main/speech_conv/metadata.jsonl"
            )
            response.raise_for_status()
            jsonl = [json.loads(line) for line in response.text.splitlines()]
        # Create 3 tasks for each entry in the metadata file.
        tasks = [TRANSCRIBE_INPUT_TASK, TRANSCRIBE_OUTPUT_TASK, ANSWER_TASK]
        self.tasks = [self._create_task(o, task) for o in jsonl for task in tasks]
        if args.num_samples:
            self.tasks = self.tasks[: args.num_samples]
        print(f"Loaded {len(self.tasks)} samples.")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if args.verbose:
            print(f"Processing sample {idx}...")
        task = self.tasks[idx]
        text = self.processor.tokenizer.apply_chat_template(
            task["messages"], tokenize=False
        )
        audio_filename = task["audio_filename"]

        # Load audio data
        if self.audio_dir:
            audio_path = os.path.join(self.audio_dir, audio_filename)
            audio, _ = librosa.load(audio_path, sr=16000)
        else:
            url = f"https://storage.googleapis.com/train-anyinstruct-speechconv-v1/{audio_filename}"
            response = self.session.get(url)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
            audio, _ = librosa.load(audio_bytes, sr=16000)
        audio = np.expand_dims(audio, axis=0)
        if args.verbose:
            print(f"Loaded audio file {audio_filename} with shape {audio.shape}")

        # Process audio and text using GazelleProcessor
        inputs = self.processor(
            text=text,
            audio=audio,
            return_tensors="pt",
            padding=True,
            # truncation=True,
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        audio_values = inputs["audio_values"].squeeze(0).to(model.device).to(dtype)

        # Create labels by shifting the input_ids to the right
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.processor.tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_values": audio_values,
            "labels": labels,
        }

    def _create_task(self, sample, task):
        chat = sample["chat"]
        if task == ANSWER_TASK:
            # We ask the LLM to generate a text answer to the input query.
            messages = [
                {"role": "user", "content": ANSWER_PROMPT},
                {"role": "assistant", "content": chat[1]["message"]},
            ]
            audio_filename = chat[0]["speech"]
        elif task == TRANSCRIBE_INPUT_TASK:
            # We ask the LLM to generate a text transcript for the input query.
            messages = [
                {"role": "user", "content": TRANSCRIBE_PROMPT},
                {"role": "assistant", "content": chat[0]["message"]},
            ]
            audio_filename = chat[0]["speech"]
        elif task == TRANSCRIBE_OUTPUT_TASK:
            # We ask the LLM to generate a text transcript for the output answer.
            messages = [
                {"role": "user", "content": TRANSCRIBE_PROMPT},
                {"role": "assistant", "content": chat[1]["message"]},
            ]
            audio_filename = chat[1]["speech"]
        return {"messages": messages, "audio_filename": audio_filename}


# Instantiate the model and processor
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id="meta-llama/Llama-2-7b-chat-hf",
)

print("Instantiating model...")
model = GazelleForConditionalGeneration(config)
print("Instantiating processor...")
llama_tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_tokenizer.pad_token_id = 0
llama_tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
model.resize_token_embeddings(len(llama_tokenizer))
processor = GazelleProcessor(
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"), llama_tokenizer
)
print("Model and processor instantiated.")

# Move the model to GPU and enable bfloat16
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Using device: {device}")
model.to(device).to(dtype)

# Prepare the dataset
train_dataset = AnyInstructSpeechDataset(args, processor)

# Set up the data loader
data_collator = DataCollatorForSeq2SeqWithAudio(tokenizer=llama_tokenizer)
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True
)

# Set up the optimizer and learning rate scheduler

lr = 2e-3
if args.optimizer == "adamw_bnb_8bit":
    optimizer = AdamW8bit(model.parameters(), lr=lr)
elif args.optimizer == "adamw":
    optimizer = AdamW(model.parameters(), lr=lr)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")
scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(step / (len(train_loader) * 0.03), 1.0)
    * math.cos(math.pi * step / len(train_loader)),
)

# Training loop
num_epochs = args.num_epochs
grad_accum_steps = 16

print("Starting training...")
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for step, batch in enumerate(train_loader):
        audio_values = batch["audio_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(
                audio_values=audio_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            print(f"Step {step + 1}/{len(train_loader)}: loss={loss.item():.4f}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
