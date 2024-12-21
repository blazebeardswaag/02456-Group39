""""
Requires a specific Python environment to run. See FID-CALC-MNIST-REQUIREMENTS.txt for the required packages.
"""

import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
from torchvision import datasets, transforms
import torch
from torch.utils import data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MNIST_MODULE = "https://www.kaggle.com/models/tensorflow/mnist/TensorFlow1/logits/1"
mnist_classifier_fn = tfhub.load(MNIST_MODULE)



def pack_images_to_tensor(path, img_size=None):
    """
    Given a path, pack all images into a tensor of shape (nb_images, height, width, channels)
    """
    path = Path(path)
    nb_images = len(list(path.rglob("*.png")))
    logger.info(f"Computing statistics for {nb_images} images")
    images = np.empty((nb_images, 28, 28, 1))
    for idx, f in enumerate(tqdm(path.rglob("*.png"))):
        img = Image.open(f)
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images[idx] = img[..., None]
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    return images_tf

def load_mnist():
    ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])
    )
    dl = data.DataLoader(ds, batch_size=60000, shuffle=False)
    x, _ = next(iter(dl))
    x = torch.permute(x, (0, 2, 3, 1))
    return tf.convert_to_tensor(x.numpy(), dtype=tf.float32)

def compute_activations(tensors, num_batches, classifier_fn):
    """
    Given a tensor of of shape (batch_size, height, width, channels), computes
    the activiations given by classifier_fn.
    """
    tensors_list = tf.split(tensors, num_or_size_splits=num_batches)
    stack = tf.stack(tensors_list)
    activation = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(classifier_fn, stack, parallel_iterations=1, swap_memory=True),
    )
    return tf.concat(tf.unstack(activation), 0)

def compute_mnist_stats(mnist_classifier_fn):
    mnist = load_mnist()
    num_batches = 1
    activations1 = compute_activations(mnist, num_batches, mnist_classifier_fn)
    return activations1

def save_activations(activations, path):
    np.save(path, activations.numpy())

def main():
    activations_path = Path("./data/mnist/activations_real.npy")
    if not activations_path.exists():
        logger.info("Computing activations for real MNIST images...")
        activations_real = compute_mnist_stats(mnist_classifier_fn)
        save_activations(activations_real, activations_path)
    else:
        logger.info("Loading precomputed activations for real MNIST images...")
        activations_real = np.load(activations_path)
        activations_real = tf.convert_to_tensor(activations_real, dtype=tf.float32)

    epoch_dir = Path('./pure_noise')
    logger.info(f"Loading images from {epoch_dir}")
    epoch_images = pack_images_to_tensor(path=epoch_dir, img_size=(28, 28))
    
    logger.info("Computing fake activations")
    activation_fake = compute_activations(epoch_images, num_batches=1, classifier_fn=mnist_classifier_fn)
    
    logger.info("Computing FID score")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        activations_real, activation_fake
    )
    logger.info(f"FID score: {fid.numpy()}")

if __name__ == "__main__":
    main()