import torch.nn as nn
import torch 
import math 


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1))
        emb = x[:, None] * emb[None, :]  
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU()
        )
        # Positional embedding added
        self.pos_emb = SinusoidalPosEmb(channels)

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(x.shape[0], self.channels, -1).swapaxes(1, 2)

        pos = self.pos_emb(torch.arange(x.shape[1], device=x.device).float())  # Shape: (seq_len, channels)
        x = x + pos[None, :, :] 

        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x 
        attention_value = self.ff_self(attention_value) + attention_value  
        return attention_value.swapaxes(2, 1).view(x.shape[0], self.channels, *size)  



class ScoreNetwork0(nn.Module):
    def __init__(self):
        super().__init__()
        nch = 4
        chs = [64, 128, 128*2, 2*256, 512, 2*512, 512]  
        # 32, 64, 128, 256, 512, 512
        # Multihead Self-Attention for 16x16
        self.attention_16x16 = MultiHeadSelfAttention(channels=chs[1], num_heads=4)
        self._convs = nn.ModuleList([
                
            # conv 0
            # 32x32
            nn.Sequential(
                nn.Conv2d(nch, chs[0], kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=chs[0]),  
                nn.ELU(alpha=1.0),
            ),
            # conv 1
            # 16x16
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=chs[1]),  
                nn.ELU(alpha=1.0),
            ),
            # conv 2
            # 8 x 8
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                nn.Dropout(p=0.1),
                nn.GroupNorm(num_groups=4, num_channels=chs[2]),  
                nn.ELU(alpha=1.0),
            ),
            # conv 3
            # 4x4
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                nn.Dropout(p=0.1),
                nn.GroupNorm(num_groups=4, num_channels=chs[3]),  
                nn.ELU(alpha=1.0),
            ),
            # conv 4
            #?
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=chs[4]),  

                nn.ELU(alpha=1.0),
            ),
            # conv 5
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(chs[4], chs[5], kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=chs[5]),  
                nn.ELU(alpha=1.0),
            ),
        ])

        # Decoder layers
        self._tconvs = nn.ModuleList([
            # dconv 0 
            nn.Sequential(
                nn.ConvTranspose2d(chs[5], chs[4], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ELU(alpha=1.0),
            ),
            # dconv 1
            nn.Sequential(
                nn.ConvTranspose2d(chs[4] * 2, chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ELU(alpha=1.0),
            ),
            # dconv 2
            nn.Sequential(
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ELU(alpha=1.0),
            ),
            # dconv 3
            nn.Sequential(
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Dropout(p=0.1),
                nn.ELU(alpha=1.0),
            ),
            # dconv 4
            nn.Sequential(
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ELU(alpha=1.0),
            ),
            # dconv 5
            nn.Sequential(
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                nn.ELU(alpha=1.0),
                nn.Conv2d(chs[0], 3, kernel_size=3, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x2 = torch.reshape(x, (*x.shape[:-1], 3, 32, 32))  # (..., 3, 32, 32)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 32, 32)  # (..., 1, 32, 32)

        
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t

        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i == 1:  
                # self-attention at the 16x16 feature map according to the paper
                # I have no idea if this is the correct way
                # but yolo
                signal = self.attention_16x16(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
              #  print(f"signal dim: {signal.shape}")
               # print(f"signals[-{i}] dim: {signals[-i].shape}")

                signal = torch.cat((signal, signals[-i]), dim=-3)  
                signal = tconv(signal)

        return signal