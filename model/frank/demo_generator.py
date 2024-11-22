import torch
import torch.optim as optim
from pathlib import Path
import sys
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frank.baseline_model import BaselineImgCaptionGen
from model.frank.data_utils.data import FlickrDataset
from model.frank.generate import CaptionGenerator
from val_utils.test_img import get_test_images
from val_utils.wandb_utils import log_image_caption
from gpt2.tokeniser import GPT2TokeniserPlus

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 40
USE_WANDB = True
CHECKPOINTS_PERIOD = 500

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

LAST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pth"

model = BaselineImgCaptionGen().to(device)

if LAST_CHECKPOINT.exists():
    model.load_state_dict(torch.load(LAST_CHECKPOINT, map_location=device))

tokeniser = GPT2TokeniserPlus()

generator = CaptionGenerator(model=model, tokeniser=tokeniser)

test_images = get_test_images(sample_count=5, seed=8315)

# --- Generate caption for unseeen images --
for img, ref_captions in test_images:
    _, generated_text = generator.generate(img)
    pass
