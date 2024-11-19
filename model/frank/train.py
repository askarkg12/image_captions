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

BATCH_SIZE = 64
EPOCHS = 5

model = BaselineImgCaptionGen().to(device)


dataset = FlickrDataset(split="train", split_size=0.1)
tokeniser = dataset.tokenizer

generator = CaptionGenerator(model=model, tokeniser=tokeniser)

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn
)

val_ds_iter = iter(
    DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
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
        exp_output = exp_output.to(device)

        preds = model(img, caption)

        filtered_preds = torch.cat([pred[:l] for pred, l in zip(preds, lens)]).to(
            device
        )

        loss = ce_loss(filtered_preds, exp_output)

        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        pass

    # Overfitted validation
    for _ in range(3):
        img, (caption, lens), exp_output = next(val_ds_iter)

        img = img.to(device)
        tkns, text = generator.generate(img)
        pass
