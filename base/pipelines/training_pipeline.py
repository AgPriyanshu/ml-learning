from constants import VALIDATION_SHAHADA_ORTHO_PATH
from trainers.deepLabV3 import DeepLabV3Trainer


if __name__ == "__main__":
    trainer = DeepLabV3Trainer(
        tile_size=1024,
        overlap=768,  # 75% overlap for smooth blending
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
