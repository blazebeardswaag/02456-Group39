import os
import numpy as np
from PIL import Image

def generate_gaussian_noise(image_shape, mean=0, std=1):
    
    return np.random.normal(mean, std, image_shape)

def add_noise_to_images(input_folder, output_folder, mean=0, std=0.02):
  
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert("RGB")

            image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            noise = generate_gaussian_noise(image_np.shape, mean, std)
            
            noisy_image = np.clip(image_np + noise, 0, 1)  # Add noise and clip to valid range [0, 1]

            noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
            noisy_image_pil = Image.fromarray(noisy_image_uint8)
            noisy_image_path = os.path.join(output_folder, filename)
            noisy_image_pil.save(noisy_image_path)

            print(f"Processed and saved: {noisy_image_path}")




def generate_and_save_gaussian_noise_images(output_folder, n=10, image_size=(32, 32), mean=0.5, std=0.2):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(n):
        noise = np.random.normal(mean, std, (image_size[1], image_size[0], 3))  
        noise = np.clip(noise, 0, 1)  #
        noise_uint8 = (noise * 255).astype(np.uint8)
        noise_image = Image.fromarray(noise_uint8)

        # Save the noise image
        image_path = os.path.join(output_folder, f'gaussian_noise_{i+1:04d}.png')
        noise_image.save(image_path)

        print(f"Generated and saved: {image_path}")

# Example Usage
if __name__ == "__main__":
    # Define output folder
    output_folder = "generated_images/cifar_noise"
    input_folder= "generated_images/cifar"
  #  add_noise_to_images(input_folder, output_folder, mean=0.5, std=0.2)



generate_and_save_gaussian_noise_images(output_folder="generated_images/noise_32x32", n = 2945, image_size=(32,32))
#input_folder, output_folder, mean=0, std=0.02
#add_noise_to_images(input_folder, output_folder)