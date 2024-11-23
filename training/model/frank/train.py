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
EPOCHS = 1000

model = BaselineImgCaptionGen().to(device)


dataset = FlickrDataset(split="train", split_size=1000)

tokeniser = dataset.tokenizer

generator = CaptionGenerator(model=model, tokeniser=tokeniser)

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn
)

val_ds= DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

ce_loss = torch.nn.CrossEntropyLoss()

optimiser = optim.AdamW(model.parameters(),lr=1e-5)

optimiser.zero_grad()

last_epoch_loss = float(100)


for epoch in range(EPOCHS):
    prog = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}, last loss: {last_epoch_loss}")
    model.train()
    losses = []
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

        losses.append(loss.item())

        optimiser.step()
        optimiser.zero_grad()

        pass

    last_epoch_loss = sum(losses)/len(losses)

    print(f"Loss: {last_epoch_loss}")

    # Overfitted validation
    model.eval()
    if epoch % 30 == 0:
        for count, batch in enumerate(val_ds):
            img, (caption, lens), exp_output = batch

            img = img.to(device)
            tkns, text = generator.generate(img)

            real_text = tokeniser.decode(exp_output)

            print(" -------============--------")
            print(f"Real text: {real_text}")
            print(f"Generated: {text}")

            if count > 5:
                break
