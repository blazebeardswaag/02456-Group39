import torch
from models.unet import ScoreNetwork0 
from sampler.image_generator import ImageGenerator
from sampler.sampler import Sampler 
from configs.config import Config 
from PIL import Image
from torchvision.transforms import Normalize
from utils.image_saver import ImageSaver

sampler = Sampler(Config, 1)
image_generator = ImageGenerator()
PATH = "model_serialzed"
model = ScoreNetwork0()
model.load_state_dict(torch.load(PATH, weights_only=True))

image_saver = ImageSaver()

# Create a list to store generated images
generated_images = []

# Generate 15 images
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