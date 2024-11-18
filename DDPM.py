import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math as math

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
testset = datasets.MNIST('data', download=True, train=False, transform=transform)

batch_size = 512
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def beta_cosine_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(0, 1, timesteps)
    return beta_start + 0.5 * (beta_end - beta_start) * (1 + torch.cos(math.pi * betas))

timesteps = 3000
betas = beta_cosine_schedule(timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# Forward Diffusion Process
def sample_forward(x0, t_int, noise):
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t_int]).view(-1, 1, 1, 1).to(device)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t_int]).view(-1, 1, 1, 1).to(device)
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

class ScoreNetwork0(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [32, 64, 128, 256, 256]
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1 + 1, chs[0], kernel_size=3, padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                nn.LogSigmoid(),
            ),
        ])
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[3]*2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[2]*2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[1]*2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.Conv2d(chs[0]*2, chs[0], kernel_size=3, padding=1),
                nn.LogSigmoid(),
                nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x2t = torch.cat((x, tt), dim=1)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=1)
                signal = tconv(signal)
        return signal
    
model = ScoreNetwork0().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training Function
def train_model(model, optimizer, train_loader, num_epochs):
    
    for epoch in range(num_epochs):
        for x0, _ in train_loader:
            x0 = x0.to(device)

            # Sample time step `t` uniformly
            t_int = torch.randint(0, timesteps, (x0.size(0),), device=device)
            t_scaled = t_int / timesteps
            t_scaled = t_scaled.unsqueeze(-1)

            # Sample noise
            noise = torch.randn_like(x0)

            # Generate noisy x_t
            noisy_x = sample_forward(x0, t_int, noise)

            # Predict noise ε_θ
            predicted_noise = model(noisy_x, t_scaled)

            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    # Save model
    torch.save(model.state_dict(), 'model.pth')

train_model(model, optimizer, train_loader, 2)

