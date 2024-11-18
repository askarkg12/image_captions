import torch
import torch.nn as nn

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frank import tapped_vit_b_16

from model.frank.decoder import get_gpt2_based_decoder


class BaselineImgCaptionGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = tapped_vit_b_16.ImageEncoder()

        self.decoder = get_gpt2_based_decoder()

        vocab_size = self.decoder.emb.num_embeddings
        emb_dim = self.decoder.emb.embedding_dim

        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, img, caption):
        pass
