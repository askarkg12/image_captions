import torch
import torch.optim as optim
from pathlib import Path
import sys
from torch.utils.data import DataLoader
import wandb
from minio import Minio

minio_client = Minio(
    "sulaiman-ov.com:9000",
    access_key="FILLKEY",
    secret_key="FILLKEY",
    secure=False,
)


# def get_checkpoint_minio():


from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frank.baseline_model import BaselineImgCaptionGen
from model.frank.data_utils.data import FlickrDataset
from model.frank.generate import CaptionGenerator
from val_utils.test_img import get_test_images
from val_utils.wandb_utils import log_image_caption

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 40
USE_WANDB = True
CHECKPOINTS_PERIOD = 500

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

LAST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pth"
minio_client.fget_object(
    "data",
    "transformer_latest.pth",
    str(LAST_CHECKPOINT),
)

model = BaselineImgCaptionGen().to(device)

if LAST_CHECKPOINT.exists():
    model.load_state_dict(torch.load(LAST_CHECKPOINT))

model.train()

train_ds = FlickrDataset(split="train")
val_ds = FlickrDataset(split="val")

tokeniser = train_ds.tokenizer

generator = CaptionGenerator(model=model, tokeniser=tokeniser)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate_fn
)

train_iter = iter(train_dl)

val_ds = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=val_ds.collate_fn
)

test_images = get_test_images(sample_count=5, seed=123)

ce_loss = torch.nn.CrossEntropyLoss()

optimiser = optim.AdamW(model.parameters(), lr=1e-5)

optimiser.zero_grad()

batch_counter = 0
epoch = 0

if USE_WANDB:
    config = {
        "model": "BaselineImgCaptionGen",
        "batch_size": BATCH_SIZE,
        "checkpoint_period": CHECKPOINTS_PERIOD,
        "lr": 1e-5,
    }
    wandb.init(project="img_caption", name="frank", config=config)


checkpoint_progress = tqdm(total=CHECKPOINTS_PERIOD, desc="Checkpioint progress")

while True:
    try:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            epoch += 1
            batch = next(train_iter)

        img, (caption, lens), exp_output = batch

        img = img.to(device)
        caption = caption.to(device)
        exp_output = exp_output.to(device)

        preds = model(img, caption)

        filtered_preds = torch.cat([pred[:l] for pred, l in zip(preds, lens)])

        loss = ce_loss(filtered_preds, exp_output)

        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        if USE_WANDB:
            wandb.log(
                {
                    "training_loss": loss.item(),
                    "epoch": epoch,
                    "batch_num": batch_counter,
                }
            )
        else:
            if batch_counter % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_counter}:")
                print(f"Training loss: {loss.item()}")

        if batch_counter % CHECKPOINTS_PERIOD == 0:
            model.eval()

            # --- Validation loss ---
            val_loss = 0
            with torch.inference_mode():
                for val_batch in val_ds:
                    val_img, (val_caption, val_lens), val_exp_output = val_batch
                    val_img = val_img.to(device)
                    val_caption = val_caption.to(device)
                    val_exp_output = val_exp_output.to(device)
                    val_preds = model(val_img, val_caption)
                    val_filtered_preds = torch.cat(
                        [pred[:l] for pred, l in zip(val_preds, val_lens)]
                    )
                    val_loss += ce_loss(val_filtered_preds, val_exp_output).item()
            val_loss /= len(val_ds)

            if USE_WANDB:
                wandb.log(
                    {"val_loss": val_loss, "epoch": epoch, "batch_num": batch_counter}
                )
            else:
                print(f"Epoch {epoch}, Batch {batch_counter}:")
                print(f"Validation Loss: {val_loss:.3f}")

            # --- Generate caption for unseeen images --
            for img_index, (img, ref_captions) in enumerate(test_images):
                _, generated_text = generator.generate(img)
                if USE_WANDB:
                    log_image_caption(
                        image=img,
                        generated_caption=generated_text,
                        reference_captions=ref_captions,
                        epoch=epoch,
                        batch_idx=batch_counter,
                        image_id=img_index,
                    )
                else:
                    print("=========================")
                    print(f"Generated: {generated_text}")
                    print(f"Reference: {ref_captions}")

            # checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{epoch}_{batch_counter}.pt"
            torch.save(model.state_dict(), LAST_CHECKPOINT)

            minio_client.fput_object(
                "data",
                "transformer_latest.pth",
                str(LAST_CHECKPOINT),
            )

            # if USE_WANDB:
            #     artifact = wandb.Artifact(
            #         "baseline_model",
            #         type="model",
            #         metadata={"epoch": epoch, "batch": batch_counter},
            #     )
            #     artifact.add_file(checkpoint_path)
            #     wandb.log_artifact(artifact)

            model.train()

            checkpoint_progress.reset()

        batch_counter += 1
        checkpoint_progress.update()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            # reduce batch size by 10%
            BATCH_SIZE = int(BATCH_SIZE * 0.9)

            train_dl = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=train_ds.collate_fn,
            )
            train_iter = iter(train_dl)
            val_ds = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=val_ds.collate_fn,
            )
        else:
            raise e
