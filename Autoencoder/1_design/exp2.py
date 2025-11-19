import torch
import torch.nn as nn

class AutoencoderExp2(nn.Module):
    """Experimento 2: Autoencoder convolucional más profundo"""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        # Encoder: (1,28,28) -> (32,14,14) -> (64,7,7) -> (128,4,4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28->28
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                     # 28->14
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 14->14
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                     # 14->7
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# 7->7
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                     # 7->3
        )

        # Decoder: 3->7->14->28 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0), # 3->7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0),   # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0, output_padding=0),    # 14->28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = AutoencoderExp2()
    x = torch.randn(1,1,28,28)
    y = model(x)
    print(y.shape)