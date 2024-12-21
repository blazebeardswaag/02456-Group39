import os
import random
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

def download_mnist_images(output_folder, n=10, seed=None):
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    transform = transforms.ToTensor()
    mnist_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    total_images = len(mnist_dataset)
    if n > total_images:
        raise ValueError(f"Requested {n} images, but MNIST only has {total_images} images.")
    selected_indices = random.sample(range(total_images), n)

    for i, idx in enumerate(selected_indices):
        image, label = mnist_dataset[idx]
        image_np = (image.squeeze().numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np, mode='L')  # MNIST images are grayscale
        
        image_path = os.path.join(output_folder, f'mnist_image_{i+1:04d}.png')
        image_pil.save(image_path)

        print(f"Saved: {image_path} (Label: {label})")


def download_cifar10_images(output_folder, n=10, seed=None):
    
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    transform = transforms.ToTensor()
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    total_images = len(cifar10_dataset)
    if n > total_images:
        raise ValueError(f"Requested {n} images, but CIFAR-10 only has {total_images} images.")

    selected_indices = random.sample(range(total_images), n)

    for i, idx in enumerate(selected_indices):
        image, label = cifar10_dataset[idx]
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # CIFAR-10 is RGB

        image_pil = Image.fromarray(image_np)
        image_path = os.path.join(output_folder, f'cifar10_image_{i+1:04d}.png')
        image_pil.save(image_path)

        print(f"Saved: {image_path} (Label: {label})")

if __name__ == "__main__":
    output_folder = "cifar10_images"

    num_images = 2945  
    download_cifar10_images(output_folder, n=num_images, seed=42)


