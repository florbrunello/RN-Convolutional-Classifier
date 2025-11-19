import torch
import torch.nn as nn


class AutoencoderExp3(nn.Module):
    """Experimento 3: Autoencoder convolucional más pequeño (menos capacidad)"""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        # Encoder más chico: (1, 28, 28) -> (16, 14, 14) -> (32, 7, 7)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (1, 28, 28) -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 28, 28) -> (16, 14, 14)
            nn.Dropout(dropout),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (16, 14, 14) -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 14, 14) -> (32, 7, 7)
        )

        # Decoder correspondiente: (32, 7, 7) -> (16, 14, 14) -> (1, 28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # (32, 7, 7) -> (16, 14, 14)
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # (16, 14, 14) -> (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Prueba rápida
if __name__ == "__main__":
    model = AutoencoderExp3()
    x = torch.randn(1,1,28,28)
    y = model(x)
    print(y.shape)
