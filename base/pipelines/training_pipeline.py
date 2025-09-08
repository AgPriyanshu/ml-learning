from segmentation_models_pytorch import DeepLabV3Plus
from src.shared.constants import IMAGENET_MEAN, IMAGENET_STD
from src.data.datasets.ortho_dataset import MultiRasterPairTileDataset
from src.models.schedulers import cosine_with_warmup
from src.models.losses import bce_loss
from src.models.optimizers import make_optimizer
from src.models.training_utils import EMA
from src.models.trainers import DeepLabV3Trainer
from src.models.losses import bce_loss
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torch.optim.lr_scheduler import LambdaLR


def run():
    # Configs.
    val_split = 0.1
    num_workers = 2
    batch_size = 14
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 30
    threshold = 0.45
    lr = 4.5e-4

    # Prepare Dataset.
    ortho_mask_pairs = [
        (
            "../SSRS/data/MOPR/ortho_cog_cropped.tif",
            "../SSRS/data/MOPR/building_mask.tif",
        ),
        (
            "../SSRS/data/Aarvi/ortho.tif",
            "../SSRS/data/Aarvi/building_mask.tif",
        ),
    ]

    dataset = MultiRasterPairTileDataset(
        ortho_mask_pairs=ortho_mask_pairs,
        tile_size=1024,
        overlap=512,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        reject_empty=True,
        train_scales=(1, 2, 4),
        augment=True,
        augment_prob=0.5,
    )

    # Prepare Loaders.
    n_total = len(dataset)
    n_val = max(1, int(val_split * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=dataset.collate_pairs,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=dataset.collate_pairs,
    )

    # Model Training.
    model = (
        DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            loss_function=bce_loss,
        )
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    optimizer = make_optimizer(
        lr=lr, model=model, weight_decay=1, backbone_lr_mult=0.5
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = LambdaLR(
        optimizer,
        cosine_with_warmup(total_steps, warmup_steps=min(1000, total_steps // 10)),
    )
    ema = EMA(model, decay=0.999)

    trainer = DeepLabV3Trainer(
        model=model,
        batch_size=batch_size,
        scheduler=scheduler,
        optimizer=optimizer,
        loss_function=bce_loss,
        device=device,
        ema=ema,
    )

    trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs)

    # trainer.load_checkpoint("./ckpts/deeplabv3p_best_overviews.pt")
    # trainer.predict(
    #     VALIDATION_SHAHADA_ORTHO_PATH,
    #     out_dir="predictions",
    #     num_workers=4,
    #     predict_scales=(1, 2),  # Multi-scale for better quality
    #     blocksize=256,  # Smaller blocks for better I/O
    # )
