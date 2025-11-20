import torch.nn as nn

class Clasificadora(nn.Module):
    def __init__(self, p, n1=128, n2=64, encoder=None):
        super().__init__()
        self._dropout = p
        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=n1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(in_features=n1, out_features=n2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(in_features=n2, out_features=10)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x