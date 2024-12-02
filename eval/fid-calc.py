import os
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow.keras import layers, models
from torchvision import datasets, transforms
import torch
from torch.utils import data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pack_images_to_tensor(path, img_size=(28, 28)):
    """
    Given a path, pack all images into a tensor of shape (nb_images, height, width, channels)
    """
    path = Path(path)
    image_files = list(path.rglob("*.png"))
    nb_images = len(image_files)
    logger.info(f"Computing statistics for {nb_images} images")
    images = []
    for idx, f in enumerate(tqdm(image_files)):
        img = Image.open(f)
        # Convert to grayscale if necessary
        if img.mode != 'L':
            img = img.convert('L')
        # Resize if not the right size
        if img_size is not None and img.size != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images.append(img[..., None])  # Add channel dimension
    images = np.stack(images, axis=0)
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    return images_tf

def load_mnist():
    ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
            ]
        ),
    )
    dl = data.DataLoader(ds, batch_size=60000, shuffle=False)
    x, y = next(iter(dl))
    x = torch.permute(x, (0, 2, 3, 1))  # Convert to (batch_size, height, width, channels)
    return tf.convert_to_tensor(x.numpy())

def compute_activations(tensors, num_batches, classifier_fn):
    """
    Given a tensor of shape (batch_size, height, width, channels), computes
    the activations given by classifier_fn.
    """
    tensors_list = tf.split(tensors, num_or_size_splits=num_batches)
    activation_list = []
    for batch in tensors_list:
        activation = classifier_fn(batch)
        activation_list.append(activation)
    activations = tf.concat(activation_list, axis=0)
    return activations

def compute_mnist_stats(mnist_classifier_fn):
    mnist = load_mnist()
    num_batches = 1
    activations1 = compute_activations(mnist, num_batches, mnist_classifier_fn)
    return activations1



def save_activations(activations, path):
    np.save(path, activations.numpy())

activations_real = np.load("./data/mnist/activations_real.npy")
activations_real = tf.convert_to_tensor(activations_real, dtype=tf.float32)

logger.info(f"Loading images of epoch {epoch_dir.name}")
epoch_images = pack_images_to_tensor(path=epoch_dir,)
logger.info("Computing fake activations")
activation_fake = compute_activations(epoch_images, 1, classifier_fn)

logger.info("Computing FID")
fid = tfgan.eval.frechet_classifier_distance_from_activations(activations_real, activation_fake)
logger.info(f"FID: {fid}")