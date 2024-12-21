import torch
import matplotlib.pyplot as plt
import os

class ImageSaver:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = os.path.join(self.project_root, "generated_images")
        os.makedirs(self.save_dir, exist_ok=True)

    def save_image_pair(self, original_img, noised_img, timestep, batch_idx):
        """
        Save original and noised images side by side
        Args:
            original_img (torch.Tensor): Image tensor (H, W)
            noised_img (torch.Tensor): Image tensor (H, W)
            timestep (int): Current timestep for this specific image
            batch_idx (int): Current batch index
        """
        # Convert to numpy arrays directly since there's no batch/channel dims
        original = original_img.cpu().numpy()  # (28, 28)
        noised = noised_img.cpu().numpy()  # (28, 28)
        
        # Denormalize from [-1, 1] to [0, 1]
        original = (original + 1) / 2
        noised = (noised + 1) / 2
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original Image', fontsize=16)
        ax1.axis('off')
        
        ax2.imshow(noised, cmap='gray')
        ax2.set_title(f'Noised Image at t={timestep}', fontsize=16)
        ax2.axis('off')
        
        # Add a main title showing batch and timestep info
        plt.suptitle(f'Batch {batch_idx}, Timestep {timestep}', fontsize=18)
        
        save_path = os.path.join(self.save_dir, f"comparison_batch{batch_idx}_t{timestep}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def save_generated_image(self, image):
        """
        Save a single generated image
        Args:
            image (torch.Tensor): Image tensor (H, W)
        """
        image = (image + 1) / 2
        
        plt.figure(figsize=(15, 15))
        plt.imshow(image.cpu().numpy(), cmap='gray')
        plt.title('Generated Image', fontsize=16)
        plt.axis('off')
        
        save_path = os.path.join(self.save_dir, "final_generated.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def save_generated_grid(self, images, n_rows=3, n_cols=5):
        """
        Save multiple generated images in a grid layout
        Args:
            images (list[torch.Tensor]): List of image tensors, each (H, W)
        """
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
        
        for idx, image in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols
            
            img_display = (image.cpu().numpy() + 1) / 2
            axes[row, col].imshow(img_display, cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle('Generated MNIST Images', fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "generated_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
  