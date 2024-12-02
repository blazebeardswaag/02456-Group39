import torch 

class ImageManager:
    def __init__(self, num_images, grid_cols=5, size=(28, 28)):
        self.num_images = num_images
        self.grid_cols = grid_cols
        self.grid_rows = (num_images + grid_cols - 1) // grid_cols
        self.size = size
        self.images = [
            {"image": torch.randn(size), "t": None, "position": idx}
            for idx in range(num_images)
        ]

    def update_image(self, idx, image, t):
        self.images[idx]["image"] = image
        self.images[idx]["t"] = t

    def get_grid(self):
        grid = []
        for row in range(self.grid_rows):
            grid_row = []
            for col in range(self.grid_cols):
                idx = row * self.grid_cols + col
                if idx < self.num_images:
                    grid_row.append(self.images[idx]["image"])
                else:
                    grid_row.append(torch.zeros(self.size))  # Empty space
            grid.append(grid_row)
        return grid

    def get_metadata(self):
        return [
            {"position": img["position"], "t": img["t"]}
            for img in self.images
        ]
