# Diffusion Models as Score Matching

## Branches

###
**main**: This branch contains scripts to sample our CIFAR-10 and MNIST using the instructions below. The models are included in main branch.

### MNIST
**patience-10**: testrun.py trains the mnist model with a patience of 10 through several combinations for a total of 48 different runs.

**patience-5**: identical to patience-10, except patience = 5.

### CIFAR-10
**new-net**: Trains cifar using gpu parallelisation, normalizes tensors for faster training among other tricks to speed up the training. testrun.py initates the training.

**modal**: same network as net-net with slightly lower learning rate using modal.com's framework to train.

all other branches are irrelavant to training or sampling from our ddpm model

## Setup

1. Create Python environment:
```bash
conda create -n fid python=3.10
conda activate fid
```

2. Install required packages:
```bash
pip install charset-normalizer==3.4.0 google-auth-oauthlib==1.2.1 tensorflow==2.15.0 tensorflow-estimator==2.15.0 tensorflow-gan==2.0.0 tensorflow-hub==0.16.1 tensorflow-intel==2.15.0 tensorflow-io-gcs-filesystem==0.31.0 tensorflow-probability==0.23.0 tensorboard==2.15.2 tensorboard-data-server==0.7.2 tf_keras==2.15.1 torch==2.5.1 torchvision==0.20.1 tqdm matplotlib opencv-python
```

## Quick Start

1. Generate images using DenoisingDiffusion:
```python
from DiffusionModel.denoising_diffusion import DenoisingDiffusion

# Create model
model = DenoisingDiffusion(task="mnist", device="cuda")

# Generate images
model.sample_images(num_images=100, output_dir="output/mnist_samples")
```

2. Calculate MNIST FID score:
```python
# Run FID calculation script
python -m eval.eval.FID-CALC-MNIST.PY
```

## Example Code

```python
# Generate and save samples
model = DenoisingDiffusion(task="mnist", device="cuda")
model.sample_images(num_images=100, output_dir="output/samples")
```

To view the samples in real-time, run the following command:
```python
model.show_diffusion(num_images=5)
```

