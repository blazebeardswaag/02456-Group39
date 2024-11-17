import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sampler.sampler import Sampler
from sampler.image_generator import ImageGenerator
from configs.config_manager import context_manager

def visualize_noise_at_t1():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    single_image = dataset[0][0].unsqueeze(0)  
    
    with context_manager() as config:
        sampler = Sampler(config, batch_size=1)
        image_generator = ImageGenerator(sampler)
        
        t = torch.ones(1, dtype=torch.long) * 4

        alpha_bar = sampler.get_alpha_bar_t(t)
        plt.figure(figsize=(20, 4))
        
        plt.subplot(1, 5, 1)
        plt.imshow(single_image[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        for i in range(4):
            eps = torch.randn_like(single_image)
            noisy_image, _ = image_generator.sample_img_at_t(t, single_image, alpha_bar, eps)
            
            plt.subplot(1, 5, i+2)
            img_display = (noisy_image[0, 0].cpu().numpy() + 1) / 2
            plt.imshow(img_display, cmap='gray')
            plt.title(f'Noised Version {i+1}')
            plt.axis('off')
        
        plt.suptitle('Different Noise Versions at t=1', fontsize=16)
        plt.savefig('noise_test_t1.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    visualize_noise_at_t1() 