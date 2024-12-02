import torch 
import numpy as np
from torchvision import datasets, transforms
from torch.utils import data
#import tensorflow as tf
#import tensorflow_gan as tfgan
from PIL import Image
import math

def sample_epsilon(xT):
    eps = torch.normal(mean=0.0, std=1.0, size=xT)
    return eps    


def get_alpha(t):
    alpha_t = 1 - linear_beta_schedueler(t)
    return alpha_t


def linear_beta_schedueler(step):
    d = (0.02 - 10**(-4))/(1000) 
    b_t = 10**(-4) + step * d 
    return b_t

def cosine_beta_scheduler(timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(0, 1, timesteps)
    return beta_start + 0.5 * (beta_end - beta_start) * (1 + torch.cos(math.pi * betas))


def get_alpha_bar_t(t):

    alpha = 1.0 - get_alpha(t)  
    alpha_bar_t = torch.cumprod(alpha, dim=0)
   # alpha_bar_t = alpha_bar_t.view(self.batch_size, 1, 1, 1)
    return alpha_bar_t

def beta_cosine_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(0, 1, timesteps)
    return beta_start + 0.5 * (beta_end - beta_start) * (1 + torch.cos(math.pi * betas))


def pack_images_to_tensor(path, img_size=None):
    """
    Given a path, pack all images into a tensor of shape (nb_images, height, width, channels)
    """
    nb_images = len(list(path.rglob("*.png")))
    logger.info(f"Computing statistics for {nb_images} images")
    images = np.empty((nb_images, 28, 28, 1))  # TODO: Consider the RGB case
    for idx, f in enumerate(tqdm(path.rglob("*.png"))):
        img = Image.open(f)
        # resize if not the right size
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
        transform=transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
            ]
        ),
    )
    dl = data.DataLoader(ds, batch_size=60000, shuffle=False)
    x, y = next(iter(dl))
    x = torch.permute(x, (0, 2, 3, 1))
    return tf.convert_to_tensor(x.numpy())

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


