from segmentation_models_pytorch import DeepLabV3Plus
from src.shared.constants import IMAGENET_MEAN, IMAGENET_STD, VALIDATION_MOPR_ORTHO_PATH, VALIDATION_SHAHADA_ORTHO_PATH
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
import os
import time


def run():
    """
    Main training pipeline with optional auto-shutdown after completion.
    
    Note: For auto-shutdown to work without password prompt, add this line to /etc/sudoers:
    YOUR_USERNAME ALL=(ALL) NOPASSWD: /sbin/shutdown
    Or run: sudo visudo and add the line above.
    """
    # Configs.
    val_split = 0.1
    num_workers = 2
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 30
    threshold = 0.65  # Increased threshold for crisper building boundaries
    lr = 2.5e-4  # Scaled for batch_size=4
    shutdown_after_training = True  # Auto-shutdown PC after training completes
    shutdown_delay = 30  # Delay in seconds before shutdown (gives time to cancel)

    # Prepare Dataset.
    # ortho_mask_pairs = [
    #     (
    #         "../SSRS/data/MOPR/ortho_cog_cropped.tif",
    #         "../SSRS/data/MOPR/building_mask.tif",
    #     ),
    #     (
    #         "../SSRS/data/Aarvi/ortho.tif",
    #         "../SSRS/data/Aarvi/building_mask.tif",
    #     ),
    # ]

    # dataset = MultiRasterPairTileDataset(
    #     ortho_mask_pairs=ortho_mask_pairs,
    #     tile_size=1024,
    #     overlap=512,
    #     mean=IMAGENET_MEAN,
    #     std=IMAGENET_STD,
    #     reject_empty=True,
    #     train_scales=(1, 2, 4),
    #     augment=True,
    #     augment_prob=0.5,
    #     min_building_ratio=0.05
    # )

    # # Prepare Loaders.
    # n_total = len(dataset)
    # n_val = max(1, int(val_split * n_total))
    # n_train = n_total - n_val
    # train_ds, val_ds = random_split(
    #     dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    # )

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     persistent_workers=(num_workers > 0),
    #     prefetch_factor=2 if num_workers > 0 else None,
    #     collate_fn=dataset.collate_pairs,
    #     drop_last=True,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=max(1, num_workers // 2),
    #     pin_memory=True,
    #     persistent_workers=(num_workers > 0),
    #     prefetch_factor=2 if num_workers > 0 else None,
    #     collate_fn=dataset.collate_pairs,
    # )

    # # Model Training.
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
    # optimizer = make_optimizer(
    #     lr=lr, model=model, weight_decay=1, backbone_lr_mult=0.5
    # )
    # total_steps = max(1, epochs * len(train_loader))
    # scheduler = LambdaLR(
    #     optimizer,
    #     cosine_with_warmup(total_steps, warmup_steps=min(1000, total_steps // 10)),
    # )
    ema = EMA(model, decay=0.999)

    trainer = DeepLabV3Trainer(
        model=model,
        batch_size=batch_size,
        loss_function=bce_loss,
        device=device,
        ema=ema,
    )

    # try:
    #     # Start training
    #     print(f"\n🚀 Starting training for {epochs} epochs...")
        # trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs, scheduler=scheduler, optimizer=optimizer)
        
    #     # Training completed successfully
    #     print("\n✅ Training completed successfully!")
        
    #     # Optional: Auto-shutdown after training
    #     if shutdown_after_training:
    #         print(f"\n🔌 System will shutdown in {shutdown_delay} seconds...")
    #         print("💡 Press Ctrl+C to cancel shutdown")
            
    #         try:
    #             # Countdown with option to cancel
    #             for i in range(shutdown_delay, 0, -1):
    #                 print(f"⏰ Shutting down in {i:2d} seconds...", end='\r')
    #                 time.sleep(1)
                
    #             print("\n🛑 Shutting down now...")
    #             # Execute shutdown command for Linux
    #             os.system('sudo shutdown -h now')
                
    #         except KeyboardInterrupt:
    #             print("\n❌ Shutdown cancelled by user")
    #             print("✅ Training results saved. System will remain on.")
        
    # except Exception as e:
    #     print(f"\n❌ Training failed with error: {e}")
    #     print("🚫 Auto-shutdown cancelled due to training failure")
    #     raise
    
    trainer.load_checkpoint("./checkpoints/deeplabv3p_best_overviews.pt")
    trainer.predict(
        VALIDATION_MOPR_ORTHO_PATH,
        out_dir="predictions",
        num_workers=8,
        predict_scales=(1, 2),  # Multi-scale for better quality
        blocksize=256,  # Smaller blocks for better I/O
        threshold=threshold,
        batch_size=8,   
    )
