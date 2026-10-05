# SRGAN Super Resolution Network

This project implements a Super-Resolution Generative Adversarial Network (SRGAN) for upscaling low-resolution images into higher-resolution outputs.

## Overview

SRGAN combines a generator and a discriminator in an adversarial setup:
- The generator learns to reconstruct high-resolution images from low-resolution inputs.
- The discriminator learns to distinguish real high-resolution images from generated ones.
- The VGG-based perceptual loss helps preserve image quality beyond pixel-wise reconstruction.

## Project structure

- `model.py` — generator and discriminator model definitions
- `dataset.py` — custom image dataset and data loading utilities
- `loss.py` — VGG perceptual loss implementation
- `utils.py` — checkpoint saving/loading and sample image generation helpers
- `train.py` — training pipeline entry point
- `config.py` — model and training configuration values
- `requirements.txt` — Python dependencies
- `images/` — sample low-resolution image assets

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses:
- PyTorch
- TorchVision
- Albumentations
- NumPy
- Pillow
- tqdm

## Setup

1. Place your training images in a folder named `new_data/` if you want to train on your own dataset.
2. If no dataset is present, the project falls back to the sample images in `images/` for a quick smoke test.
3. Adjust training hyperparameters in `config.py` as needed.

## Training

Run:

```bash
python train.py
```

This will:
- load the dataset,
- create the generator and discriminator,
- train for the configured number of epochs,
- save checkpoints as `gen.pth.tar` and `disc.pth.tar`.

## Notes

- The training loop is intentionally lightweight for validation/debugging and can be expanded for full training runs.
- VGG weights are downloaded automatically on first use from PyTorch model hub.
- CPU training works, but GPU is strongly recommended for faster training.

## Example usage

After training, you can adapt the project to run inference on a specific image by loading the saved generator and passing a low-resolution image through it.

## License

This project is provided for educational and experimental use.
