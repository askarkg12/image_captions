import torch
import torch.optim as optim
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frank.baseline_model import BaselineImgCaptionGen
from model.frank.data_utils.data import FlickrDataset
from model.frank.generate import CaptionGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 48

VAL_PERIOD = 30


model = BaselineImgCaptionGen().to(device)


train_ds = FlickrDataset(split="train")
val_ds = FlickrDataset(split="val")

tokeniser = train_ds.tokenizer

generator = CaptionGenerator(model=model, tokeniser=tokeniser)

dataloader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate_fn
)

val_ds = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=val_ds.collate_fn
)

ce_loss = torch.nn.CrossEntropyLoss()

optimiser = optim.AdamW(model.parameters(), lr=1e-5)

optimiser.zero_grad()

while True:
    pass
