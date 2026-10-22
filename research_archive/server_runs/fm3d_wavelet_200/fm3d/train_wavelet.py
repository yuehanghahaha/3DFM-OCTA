"""Train the 3D Haar flow experiment: 200 epochs, snapshots every five."""
from .train import main


if __name__ == '__main__':
    main(default_representation='haar3d', default_epochs=200, default_save_every=5)
