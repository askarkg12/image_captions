import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from gpt2.tokeniser import GPT2TokeniserPlus

# Load model directly
from transformers import AutoModelForCausalLM


class FrankensteinCaption(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int | None = None,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        if ff_dim is None:
            ff_dim = embed_dim * 4

        super().__init__()

        self.emb = nn.Embedding(vocab_size, embed_dim)

        self.pos_emb = SinusoidalPositionalEmbeddings(embed_dim)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
            ),
            num_layers,
        )

    def forward(self, x, memory):
        x = self.emb(x) + self.pos_emb(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )
        x = self.transformer(
            x,
            memory,
            tgt_is_causal=True,
            tgt_mask=causal_mask.repeat(memory.size(0), 1, 1),
        )
        return x


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # Ensure that positions dont start at 0
        position = torch.arange(1, max_len + 1).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, d_model, 2) * (torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(-1)
        return self.pe[:seq_len].unsqueeze(0)


def get_gpt2_based_decoder(num_layers=2):
    tokenizer = GPT2TokeniserPlus()

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    wte = model.transformer.wte.weight.data

    decoder = FrankensteinCaption(
        vocab_size=tokenizer.vocab_size,
        embed_dim=768,
        num_layers=num_layers,
        num_heads=12,
    )

    decoder.emb.weight.data[: wte.shape[0]] = wte

    return decoder


if __name__ == "__main__":
    decoder = get_gpt2_based_decoder()

    pass
