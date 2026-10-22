"""Train voxel FM with source-only Z projection supervision, 256x96x96, 200 epochs."""
from .train import main


if __name__ == '__main__':
    main(default_epochs=200, default_save_every=5, default_patch=(256, 96, 96),
         default_lambda_projection=0.1)
