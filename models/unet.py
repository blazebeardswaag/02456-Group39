import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels], eps = 1e-2)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels], eps = 1e-2),
            nn.Linear(channels, channels),
            nn.LogSigmoid()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(x.shape[0], self.channels, -1).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(x.shape[0], self.channels, *size)

class ScoreNetwork0(nn.Module):
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 512]
        
        # Encoder layers
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, chs[0], kernel_size=3, padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                nn.LogSigmoid(),
                SelfAttention(chs[1]),  # Added attention at 14x14 resolution
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

        # Decoder layers
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LogSigmoid(),
                SelfAttention(chs[1]),  # Added attention at 14x14 resolution
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                nn.LogSigmoid(),
                nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.parameters()).device)
        t = t.to(next(self.parameters()).device)
        
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        
        # Encoder forward pass
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        # Decoder forward pass
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
                
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))
        return signal