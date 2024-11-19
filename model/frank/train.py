import torch
import torch.optim as optim
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frank.baseline_model import BaselineImgCaptionGen
from model.frank.data_utils.data import FlickrDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 5

model = BaselineImgCaptionGen().to(device)


dataset = FlickrDataset(split="train", split_size=0.1)
tokeniser = dataset.tokenizer

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn
)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn
)

ce_loss = torch.nn.CrossEntropyLoss()

optimiser = optim.AdamW(model.parameters())

optimiser.zero_grad()


for epoch in range(EPOCHS):
    prog = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in prog:
        img, (caption, lens), exp_output = batch

        img = img.to(device)
        caption = caption.to(device)

        preds = model(img, caption)

        filtered_preds = torch.cat([pred[:l] for pred, l in zip(preds, lens)])

        loss = ce_loss(filtered_preds, exp_output)

        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        pass

    # Overfitted validation
    sample = dataset[random.randint(0, len(dataset))]
