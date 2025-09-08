from constants import VALIDATION_SHAHADA_ORTHO_PATH
from src.models.trainers import DeepLabV3Trainer


if __name__ == "__main__":
    if ortho_mask_pairs is None:
        if image_tif_path is not None and label_tif_path is not None:
            # Legacy single-file mode
            print(
                "Warning: Using legacy single-file mode. Consider using ortho_mask_pairs parameter."
            )
            ortho_mask_pairs = [(image_tif_path, label_tif_path)]
        else:
            # Default to MOPR and Aarvi datasets
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
            print("Using default MOPR and Aarvi datasets")

    dataset = MultiRasterPairTileDataset(
        ortho_mask_pairs=ortho_mask_pairs,
        tile_size=self.tile_size,
        overlap=self.overlap,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        reject_empty=reject_empty,
        train_scales=(1, 2, 4),
        augment=augment,
        augment_prob=augment_prob,
    )
    n_total = len(dataset)
    n_val = max(1, int(val_split * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # High-throughput DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=self.batch_size,
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
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=dataset.collate_pairs,
    )

    model = (
        DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        .to(self.device)
        .to(memory_format=torch.channels_last)
    )

    trainer = DeepLabV3Trainer(
        batch_size=8,
        threshold=0.49,
        blending_method="distance",  # Use distance-based blending
        border_fade=128,  # Fade 128 pixels from edges
    )

    # Train with default MOPR and Aarvi datasets (will be used automatically)
    # trainer.train(
    #     epochs=30,
    #     reject_empty=True,
    #     num_workers=4,
    #     augment=True,
    #     augment_prob=0.6,
    # )

    trainer.load_checkpoint("./ckpts/deeplabv3p_best_overviews.pt")
    trainer.predict_streaming(
        VALIDATION_SHAHADA_ORTHO_PATH,
        out_dir="predictions",
        num_workers=4,
        predict_scales=(1, 2),  # Multi-scale for better quality
        blocksize=256,  # Smaller blocks for better I/O
    )
