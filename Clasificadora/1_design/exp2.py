import torch.nn as nn

class Clasificadora2(nn.Module):
    def __init__(self, p, n1, n2, n3, encoder=None):
        super().__init__()
        self._dropout = p

        self.encoder = encoder
        self.classifier = nn.Sequential(
            # Capa 0
            nn.Flatten(),
            
            # Capa 1
            nn.Linear(in_features = 64*7*7, out_features = n1),
            nn.ReLU(),
            nn.Dropout(p),
            
            # Capa 2
            nn.Linear(in_features = n1, out_features = n2),
            nn.ReLU(),
            nn.Dropout(p),
            
            # Capa 2
            nn.Linear(in_features = n2, out_features = n3),
            nn.ReLU(),
            nn.Dropout(p),

            # Capa 3
            nn.Linear(in_features = n3, out_features = 10)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x