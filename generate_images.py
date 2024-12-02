import os
import torch
from models.unet import ScoreNetwork0
from sampler.image_generator import ImageGenerator
from sampler.sampler import Sampler
from configs.config import Config
from configs.config_manager import context_manager
from PIL import Image


def load_model(device, model_path="model_serialzed"):
    model = ScoreNetwork0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def initialize_image(size=(28, 28)):
    return torch.randn(size)


def generate_single_image(image_idx, model, sampler, x_t):
    image_generator = ImageGenerator()
    with torch.no_grad():
        for t in range(1000, 1, -1):
            print(f"Constructing image {image_idx + 1} at timestep: {t}", end="\r")
            t_tensor = torch.tensor([t]).unsqueeze(0)

            # Calculate required values
            eps_theta = model(x_t.view(1, -1), t_tensor)
            alpha_t = sampler.get_alpha(t_tensor)
            alpha_bar_t = sampler.get_alpha_bar_t(t_tensor)
            beta_t = sampler.linear_beta_schedueler(t_tensor)
            z = torch.randn_like(x_t) if t > 1 else 0

            x_t = image_generator.reconstruct_image(
                x_t.view(28, 28),
                eps_theta.view(28, 28),
                t_tensor,
                alpha_t,
                alpha_bar_t,
                beta_t,
                z,
            )
            x_t = x_t.view(28, 28)

    return x_t


def save_image(tensor_image, output_path):
    tensor_image = ((tensor_image + 1) / 2 * 255).clamp(0, 255).byte()
    tensor_image = tensor_image.cpu().numpy()
    image = Image.fromarray(tensor_image, mode="L")  # Grayscale image
    image.save(output_path)


def generate_and_save_images(output_dir, sampler, config, num_images=15):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(config.device)
    for image_idx in range(num_images):
        print(f"\nGenerating image {image_idx + 1}/{num_images}")
        x_t = initialize_image()
        generated_image = generate_single_image(image_idx, model, sampler, x_t)

        # Save each image
        output_path = os.path.join(output_dir, f"image_{image_idx + 1}.png")
        save_image(generated_image, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_directory = "eval/generated_images"
    with context_manager(
        batch_size=1000,
        LR=1e-4,
        experiment_name="mnist_training",
        scheduler_type="linear",
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ) as config:
        sampler = Sampler(config, 1)
        generate_and_save_images(output_directory, sampler, config, num_images=100)
