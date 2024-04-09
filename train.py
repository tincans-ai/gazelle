import argparse
import io
import math
import os
from dataclasses import dataclass
from datetime import datetime

import librosa
import numpy as np
import requests
import torch
from bitsandbytes.optim import AdamW8bit
from datasets import Audio, Dataset, IterableDataset, load_dataset
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, LlamaTokenizerFast, Wav2Vec2Processor

from gazelle import GazelleConfig, GazelleForConditionalGeneration, GazelleProcessor

SAMPLE_RATE = 16000
AUDIO_MODEL = "facebook/wav2vec2-base-960h"
TEXT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

TRANSCRIBE_INPUT_TASK = "transcribe_input"
TRANSCRIBE_OUTPUT_TASK = "transcribe_output"
ANSWER_TASK = "answer"

TRANSCRIBE_PROMPTS = [
    # from Gazelle
    "Transcribe <|audio|>",
    "Transcribe exactly what is said here <|audio|>",
    "Repeat exactly what is written here: <|audio|>",
    "Write exactly what was said: <|audio|>",
    "First listen to the clip. Then, transcribe exactly what is said. <|audio|>",
    # from GPT-4
    "Capture every word from <|audio|> verbatim",
    "Convert speech to text from <|audio|>",
    "Listen and transcribe the complete text from <|audio|>",
    "Record in writing what is spoken in <|audio|>",
    "Transcribe the spoken words from <|audio|> with exact wording and punctuation",
]
ANSWER_PROMPTS = [
    # from Gazelle
    "Listen to <|audio|> and respond to it",
    "Listen and respond: <|audio|>",
    "Respond to <|audio|>",
    "Respond to the user <|audio|>",
    "Respond to this question: \n<|audio|>",
    "Continue the conversation after <|audio|>",
    "First listen to the clip: <|audio|>\n How would you respond?",
    "<|audio|> – respond",
    "<|audio|>\n Respond to the question",
    "<|audio|>",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-set", "-d", type=str, default="boolq", choices=["boolq", "anyinstruct"]
)
parser.add_argument("--data-dir", "-p", type=str)
parser.add_argument("--device", "-D", type=str, default="cuda")
parser.add_argument("--data-type", "-t", type=str, default="bfloat16")
parser.add_argument("--optimizer", "-o", type=str, default="adamw_bnb_8bit")
parser.add_argument("--num-prompts", "-P", type=int, default=1)
parser.add_argument("--num-samples", "-n", type=int)
parser.add_argument("--num-epochs", "-e", type=int, default=1)
parser.add_argument("--grad-accum-steps", "-g", type=int, default=16)
parser.add_argument("--batch-size", "-b", type=int, default=4)
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


class GazelleDataset(IterableDataset):
    def __init__(self, dataset, processor, args):
        self.dataset = dataset
        self.processor = processor
        self.args = args
        self.session = requests.Session()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if args.verbose:
            print(f"Processing sample {idx}...")
        messages, audio = self._get_data(idx, self.dataset[idx])
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False)
        return self._make_audio_sample(text, audio)

    def select(self, indices):
        self.dataset = self.dataset.select(indices)
        return self

    def _get_answer_prompt(self, idx):
        prompt_idx = idx % min(self.args.num_prompts, len(ANSWER_PROMPTS))
        return ANSWER_PROMPTS[prompt_idx]

    def _get_transcribe_prompt(self, idx):
        prompt_idx = idx % min(self.args.num_prompts, len(TRANSCRIBE_PROMPTS))
        return TRANSCRIBE_PROMPTS[prompt_idx]

    def _load_audio(self, base_url: str, folder: str, filename: str):
        if self.args.data_dir:
            audio_path = f"{self.args.data_dir}/{folder}/{filename}"
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        else:
            url = f"{base_url}/{filename}"  # hack for GCS bucket naming
            response = self.session.get(url)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
            audio, _ = librosa.load(audio_bytes, sr=SAMPLE_RATE)
        return audio

    def _make_audio_sample(self, text: str, audio: np.ndarray):
        # Process audio and text using GazelleProcessor.
        # Audio is expanded to be a [C x M] array, although C=1 for mono audio.
        inputs = self.processor(
            text=text,
            audio=np.expand_dims(audio, axis=0),
            return_tensors="pt",
            text_padding=True,
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        audio_values = inputs["audio_values"].squeeze(0).to(model.device).to(dtype)

        # Create labels by shifting the input_ids to the right
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = processor.tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_values": audio_values,
            "labels": labels,
        }


class AnyInstructSpeechDataset(GazelleDataset):
    """
    Metadata file format:
    {"chat": [
        {"role": "USER", "message": "Write a sentence based on this summary: iraqi embassy in jakarta removes saddam hussein 's photo", "speech": "chunk_00000/0001.mp3"},
        {"role": "AnyGPT", "message": "The building in Jakarta where people from Iraq work, took down a picture of a man named Saddam Hussein.", "speech": "chunk_00000/0002.mp3"}
    ]}
    """

    def __init__(self, args, processor):
        dataset = load_dataset(
            "json",
            data_dir=args.data_dir,
            data_files="https://huggingface.co/datasets/fnlp/AnyInstruct/resolve/main/speech_conv/metadata.jsonl",
            split="train",
        )
        super().__init__(dataset, processor, args)

    def _get_data(self, idx, row):
        return self._create_task(idx, row, ANSWER_TASK)

    def _create_task(self, idx, sample, task):
        chat = sample["chat"]
        if task == ANSWER_TASK:
            # We ask the LLM to generate a text answer to the input query.
            messages = [
                {"role": "user", "content": self._get_answer_prompt(idx)},
                {"role": "assistant", "content": chat[1]["message"]},
            ]
            audio_filename = chat[0]["speech"]
        elif task == TRANSCRIBE_INPUT_TASK:
            # We ask the LLM to generate a text transcript for the input query.
            messages = [
                {"role": "user", "content": self._get_transcribe_prompt(idx)},
                {"role": "assistant", "content": chat[0]["message"]},
            ]
            audio_filename = chat[0]["speech"]
        elif task == TRANSCRIBE_OUTPUT_TASK:
            # We ask the LLM to generate a text transcript for the output answer.
            messages = [
                {"role": "user", "content": self._get_transcribe_prompt(idx)},
                {"role": "assistant", "content": chat[1]["message"]},
            ]
            audio_filename = chat[1]["speech"]

        speech_base_url = (
            "https://storage.googleapis.com/train-anyinstruct-speechconv-v1"
        )
        audio = self._load_audio(speech_base_url, "speech", audio_filename)
        return messages, audio


class BoolQDataset(GazelleDataset):
    def __init__(self, args, processor):
        dataset = Dataset.load_from_disk(os.path.join(args.data_dir)).cast_column(
            "audio", Audio(sampling_rate=SAMPLE_RATE)
        )
        super().__init__(dataset, processor, args)

    def _get_data(self, idx, row):
        messages = [
            {"role": "user", "content": self._get_answer_prompt(idx)},
            {"role": "assistant", "content": row["question"]},
        ]
        return messages, row["audio"]["array"]


# Instantiate the model and processor
config = GazelleConfig(
    audio_model_id=AUDIO_MODEL,
    text_model_id=TEXT_MODEL,
)

print("Instantiating model...")
model = GazelleForConditionalGeneration(config)
print("Instantiating processor...")
text_tokenizer = LlamaTokenizerFast.from_pretrained(TEXT_MODEL)
text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
model.resize_token_embeddings(len(text_tokenizer))
audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL)
processor = GazelleProcessor(audio_processor, text_tokenizer)
print("Model and processor instantiated.")

# Move the model to GPU and enable bfloat16
device_type = (
    args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
)
device = torch.device(device_type)
print(f"Using dtype and device: {dtype}, {device}")
model.to(device).to(dtype)

# Prepare the dataset
if args.data_set == "anyinstruct":
    train_dataset = AnyInstructSpeechDataset(args, processor)
elif args.data_set == "boolq":
    train_dataset = BoolQDataset(args, processor)
else:
    raise ValueError(f"Unknown dataset: {args.data_set}")
if args.num_samples:
    train_dataset = train_dataset.select(range(args.num_samples))
print(f"Loaded {len(train_dataset)} samples.")

# Set up the data loader
data_collator = DataCollatorForSeq2SeqWithAudio(tokenizer=text_tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=data_collator,
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
print("Starting training...")
num_epochs = args.num_epochs
grad_accum_steps = args.grad_accum_steps
print(f"epochs: {num_epochs}")
print(f"grad_accum_steps: {grad_accum_steps}")
print(f"learning_rate: {lr}")
print(f"train dataset size: {len(train_dataset)}")
print(f"train batchsize: {args.batch_size}")
t_start = datetime.now()
print(f"start time: {t_start}")

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

t_end = datetime.now()
print(f"end time: {t_end}")
print(f"elapsed: {t_end - t_start}")
# print(f"total audio: {train_dataset.total_audio} seconds")

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving model to {output_dir}")
model.save_pretrained(output_dir, from_pt=True)
text_tokenizer.save_pretrained(output_dir, from_pt=True)
audio_processor.save_pretrained(output_dir, from_pt=True)
