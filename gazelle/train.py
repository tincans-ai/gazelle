import math
import torch
import librosa
from torch.utils.data import DataLoader, Dataset
from modeling_gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

AUDIO_FILES = []
TEXT_DATA = []


class SpeechDataset(Dataset):
    def __init__(self, audio_files, text_data, processor):
        self.audio_files = audio_files
        self.text_data = text_data
        self.processor = processor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        text = self.text_data[idx]

        # Load audio data
        audio, _ = librosa.load(audio_file, sr=16000)

        # Process audio and text using GazelleProcessor
        inputs = self.processor(
            text=text, audio=audio, return_tensors="pt", padding=True, truncation=True
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        audio_values = inputs["audio_values"].squeeze(0)

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


# Instantiate the model and processor
config = GazelleConfig()  # stock options
model = GazelleForConditionalGeneration(config)
processor = GazelleProcessor()

# Move the model to GPU and enable bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.to(torch.bfloat16)

# Prepare the dataset
train_dataset = SpeechDataset(AUDIO_FILES, TEXT_DATA, processor)

# Set up the data loader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Set up the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(step / (len(train_loader) * 0.03), 1.0)
    * math.cos(math.pi * step / len(train_loader)),
)

# Training loop
num_epochs = 1
grad_accum_steps = 16

model.train()
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        audio_values = batch["audio_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
