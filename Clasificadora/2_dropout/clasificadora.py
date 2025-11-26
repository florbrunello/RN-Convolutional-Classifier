import torch.nn as nn

class Clasificadora(nn.Module):
    def __init__(self, p, encoder=None):
        super().__init__()
        self._dropout = p

        self.encoder = encoder
        self.classifier = nn.Sequential(
            # Capa 0
            nn.Flatten(),
            
            # Capa 1
            nn.Linear(in_features = 64*7*7, out_features = 512),
            nn.ReLU(),
            nn.Dropout(p),
            
            # Capa 2
            nn.Linear(in_features = 512, out_features = 256),
            nn.ReLU(),
            nn.Dropout(p),

            # Capa 3
            nn.Linear(in_features = 256, out_features = 10)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x