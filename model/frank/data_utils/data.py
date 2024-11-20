import torch
import torchvision
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import sentencepiece as spm

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gpt2.tokeniser import GPT2TokeniserPlus


class FlickrDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        split_size: int | float = None,
        cache_path: str | Path | None = None,
    ):
        self.tokenizer = GPT2TokeniserPlus()
        # Padding token is outside of vocab

        self.img_to_tensor = (
            torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        )
        if cache_path is None:
            cache_path = Path(__file__).parent / f"flickr_cache_{split}.pkl"

        if cache_path is not None:
            if os.path.exists(cache_path):
                data = pickle.load(open(cache_path, "rb"))
                self.id_to_img, self.img_caption_pairs = data
                return
            else:
                print(f"Cache file not found at {cache_path}, re-creating data")

        if isinstance(split_size, float):
            if split_size > 1 or split_size < 0:
                raise ValueError("split_size should be between 0 and 1, if float")
            ratio = split_size
        hg_ds = load_dataset(
            "nlphuji/flickr30k",
            split="test",
        ).filter(lambda x: x["split"] == split)

        if isinstance(split_size, int):
            ratio = split_size / len(hg_ds)
        if split_size is not None:
            # Fake random sampling of subset
            hg_ds = hg_ds.train_test_split(test_size=ratio)["test"]
        self.id_to_img = {
            int(row["img_id"]): self.img_to_tensor(row["image"])
            for row in tqdm(hg_ds, desc="Preprocessing images")
        }

        self.img_caption_pairs: list[tuple[int, str]] = []
        for row in hg_ds:
            pairs = [(int(row["img_id"]), caption) for caption in row["caption"]]
            self.img_caption_pairs.extend(pairs)

        if cache_path is not None:
            with open(cache_path, "wb") as f:
                data = (self.id_to_img, self.img_caption_pairs)
                pickle.dump(data, f)
                print(f"Data saved to {cache_path}")

    # def resize_to_width(self, img: Image.Image, width: int):
    #     wpercent = width / float(img.size[0])
    #     hsize = int((float(img.size[1]) * float(wpercent)))
    #     return img.resize((width, hsize), Image.Resampling.LANCZOS)

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, idx) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        img_id, caption = self.img_caption_pairs[idx]

        tokens = torch.tensor(
            self.tokenizer.encode(caption, add_bos=True, add_eos=True)
        )
        return (
            self.id_to_img[img_id],
            tokens[:-1],
            tokens[1:],
        )

    # def as_flat_tensors(self, img: Image.Image, box_width: int = 16):
    #     data: torch.Tensor = self.img_to_tensor(img)
    #     patches = torch.nn.functional.unfold(
    #         data.unsqueeze(0),
    #         kernel_size=(box_width, box_width),
    #         stride=(box_width, box_width),
    #     )
    #     return patches.squeeze(0).T

    def collate_fn(self, batch):
        imgs, in_caption, out_caption = zip(*batch)

        # img_lens = torch.tensor([img.size(0) for img in imgs])
        # img_tnsr = pad_sequence(imgs, batch_first=True)
        img_tnsr = torch.stack(imgs)

        # No need to calculate twice
        caption_lens = [len(caption) for caption in in_caption]
        in_tnsr = pad_sequence(
            in_caption, batch_first=True, padding_value=self.tokenizer.pad_id
        )
        out_tnsr = torch.cat(out_caption, dim=0)

        return img_tnsr, (in_tnsr, caption_lens), out_tnsr


if __name__ == "__main__":
    dataset = FlickrDataset(split="val", split_size=10)
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dl:
        pass
