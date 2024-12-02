import torch
from models.unet import ScoreNetwork0 
from sampler.image_generator import ImageGenerator
from sampler.sampler import Sampler 
from configs.config import Config 
from PIL import Image
from torchvision.transforms import Normalize
from utils.image_saver import ImageSaver
from configs.config_manager import context_manager



def load_model(device, model_path="model_serialzed"):

    model = ScoreNetwork0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def initialize_image(size=(28, 28)):

    return torch.randn(size)


def generate_one_step(model, sampler, t,image_generator ):
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


"""def generate_images(sampler, config, num_images=15):
 
    model = load_model(config.device)
    image_saver = ImageSaver()
    generated_images = []

    for image_idx in range(num_images):
        print(f"\nGenerating image {image_idx + 1}/{num_images}")
        x_t = initialize_image()
        generated_image = generate_single_image(
            image_idx, model, sampler, x_t
        )
        generated_images.append(generated_image)

    # Save all generated images in a grid
    image_saver.save_generated_grid(generated_images)
"""

def generate_image(sampler, config):
    image_generator = ImageGenerator()
    PATH = "model_serialzed"
    model = ScoreNetwork0().to(config.device)
    model.load_state_dict(torch.load(PATH, map_location=config.device))

    image_saver = ImageSaver()

    generated_images = []

    for image_idx in range(15):
        print(f"\nGenerating image {image_idx + 1}/15")
        
        SIZE = torch.zeros(28,28)
        x_t = torch.randn_like(SIZE)
        t_1000 = x_t.clone()
        x_t = (x_t + 1) / 2

        with torch.no_grad():
            for t in range(1000, 1 , -1):
                print(f"constructing image {image_idx + 1} at timestep: {t}", end='\r')
                t_tensor = torch.tensor([t]).unsqueeze(0)
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
                    z)
                x_t = x_t.view(28,28)
            
            generated_images.append(x_t)

    # Save all generated images in a grid
    image_saver.save_generated_grid(generated_images)


with context_manager(  
    batch_size=1000,
    LR=1e-4,
    experiment_name="mnist_training",
    scheduler_type="linear",
    device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as config:
    sampler = Sampler(config, 1)
    generate_image(sampler, config)
 