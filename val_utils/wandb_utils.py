import wandb
import torch
from PIL import Image
import numpy as np


def log_image_caption(
    *, image, generated_caption, reference_captions, epoch, batch_idx, image_id=None
):
    """
    Log image caption progress to WandB.

    Args:
        image: PIL Image or tensor
        generated_caption (str): Model generated caption
        reference_captions (list): List of ground truth/reference captions
        epoch (int): Current training epoch
        batch_idx (int): Current batch index
        image_id (str, optional): Unique identifier for the image
    """

    # Create caption comparison table
    caption_table = wandb.Table(columns=["Type", "Caption"])
    caption_table.add_data("Generated", generated_caption)
    for idx, ref_cap in enumerate(reference_captions):
        caption_table.add_data(f"Reference {idx+1}", ref_cap)

    # Create a unique identifier for the image if not provided
    if image_id is None:
        image_id = f"img_{epoch}_{batch_idx}"

    # Log everything to WandB
    wandb.log(
        {
            f"caption_evolution/{image_id}": wandb.Image(
                image,
                caption=generated_caption,
                masks={
                    "predictions": {
                        "mask_data": None,  # No mask for this example
                        "class_labels": {
                            "Generated": generated_caption,
                            **{
                                f"Reference {i+1}": cap
                                for i, cap in enumerate(reference_captions)
                            },
                        },
                    }
                },
            ),
            f"captions_table/{image_id}": caption_table,
            "epoch": epoch,
            "batch": batch_idx,
        }
    )
