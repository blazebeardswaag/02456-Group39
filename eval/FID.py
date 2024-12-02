"""import numpy as np
import logger 
from torchvision import datasets, transforms
from torch.utils import data
import tensorflow as tf
import tensorflow_gan as tfgan
from utils.helpers import pack_images_to_tensor, compute_activations



def evaluate_fid(epoch_dir, classifier_fn):
        
    activations_real = np.load("./data/mnist/activations_real.npy")
    activations_real = tf.convert_to_tensor(activations_real, dtype=tf.float32)
    logger.info(f"Loading images of epoch {epoch_dir.name}")
    epoch_images = pack_images_to_tensor(
            path=epoch_dir,
        )
    logger.info("Computing fake activations")
    activation_fake = compute_activations(epoch_images, 1, classifier_fn)

    logger.info("Computing FID")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        activations_real, activation_fake
    )
    logger.info(f"FID: {fid}")"""