from pathlib import Path
import sys
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frank.model import BaselineImgCaptionGen
from model.frank.data_utils.data import FlickrDataset

BATCH_SIZE = 16

model = BaselineImgCaptionGen()


dataset = FlickrDataset(split="val", split_size=10)
tokeniser = dataset.tokenizer

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn
)
