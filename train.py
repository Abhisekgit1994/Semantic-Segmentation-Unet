import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet34
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import segmentation_models_pytorch as smp
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# resnet = resnet34(pretrained=True)

# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 368 originally
IMAGE_WIDTH = 256  # 368 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/tissue images"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMG_DIR = "data/val/tissue images"
VAL_MASK_DIR = "data/val/masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    # check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print("epoch:", epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer":optimizer.state_dict(),
        # }
        # save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # if dice >= 0.76:
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     save_checkpoint(checkpoint, dice)

        # # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )


if __name__ == "__main__":
    main()