import torch.nn as nn


class FCLayer(nn.Module):
    # Fully Connected Layer
    def __init__(self):
        super(FCLayer, self).__init__()
        self.batch = 0

        # Fully Connected Layer
        self.fcLayer = nn.Sequential(
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        self.batch = x.shape[0]
        x = x.view(self.batch, -1)
        x = self.fcLayer(x)
        return x
