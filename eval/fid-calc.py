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

def save_activations(activations, path):
    np.save(path, activations.numpy())

def create_mnist_classifier_for_training():
    """
    Creates and returns a CNN model for training on MNIST.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(64, 3, strides=2, activation='relu'),
        layers.Conv2D(128, 3, strides=2, activation='relu'),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(128, activation='relu'),  # Activations for FID
        layers.Dense(10, activation='softmax')  # Classification output
    ])
    return model

def create_mnist_classifier_for_activations():
    """
    Creates and returns a CNN model for computing activations.
    """
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, 3, strides=2, activation='relu')(inputs)
    x = layers.Conv2D(128, 3, strides=2, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    activations = layers.Dense(128, activation='relu')(x)  # Activations for FID
    model = models.Model(inputs=inputs, outputs=activations)
    return model

def train_mnist_classifier():
    """
    Trains the classifier on the MNIST dataset and saves the weights.
    """
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)

    model = create_mnist_classifier_for_training()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.save_weights('mnist_classifier_weights.h5')

def classifier_fn(images):
    """
    Classifier function to compute activations for FID calculation.
    """
    model = create_mnist_classifier_for_activations()
    model.load_weights('mnist_classifier_weights.h5', by_name=True)
    activations = model(images)
    return activations

def main():
    if not os.path.exists('mnist_classifier_weights.h5'):
        logger.info('Training MNIST classifier...')
        train_mnist_classifier()
    else:
        logger.info('MNIST classifier weights found.')

    if not os.path.exists('./data/mnist/activations_real.npy'):
        logger.info('Computing activations of real images...')
        activations_real = compute_activations(load_mnist(), num_batches=1, classifier_fn=classifier_fn)
        save_activations(activations_real, './data/mnist/activations_real.npy')
    else:
        logger.info('Loading activations of real images...')
        activations_real = np.load('./data/mnist/activations_real.npy')
        activations_real = tf.convert_to_tensor(activations_real, dtype=tf.float32)

    epoch_dir = './generated_images'
    logger.info(f"Loading images from {epoch_dir}")
    epoch_images = pack_images_to_tensor(path=epoch_dir)
    logger.info("Computing activations of generated images")
    activation_fake = compute_activations(epoch_images, num_batches=1, classifier_fn=classifier_fn)

    logger.info("Computing FID")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        activations_real, activation_fake
    )
    logger.info(f"FID: {fid.numpy()}")

if __name__ == '__main__':
    main()