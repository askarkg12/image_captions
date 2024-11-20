from datasets import load_dataset
from PIL.Image import Image


def get_test_images(
    *,
    indices: list[int] | None = None,
    sample_count: int | None = None,
    seed: int | None = None
) -> list[tuple[Image, list[str]]]:
    hg_ds = load_dataset(
        "nlphuji/flickr30k",
        split="test",
    ).filter(lambda x: x["split"] == "test")

    if indices is None:
        if sample_count is None:
            raise ValueError("Either indices or sample_count must be provided")
        if seed is not None:
            hg_ds = hg_ds.shuffle(seed=seed)
        return [(hg_ds[i]["image"], hg_ds[i]["caption"]) for i in range(sample_count)]

    return [(hg_ds[i]["image"], hg_ds[i]["caption"]) for i in indices]


if __name__ == "__main__":
    a = get_test_images(sample_count=5, seed=12)
    b = get_test_images(sample_count=5, seed=12)
    pass
