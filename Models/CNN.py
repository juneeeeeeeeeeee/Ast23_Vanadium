import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.batch = 0
        self.layer = nn.Sequential(
            # batch x 3 x 32 x 32 -> batch x 64 x 32 x 32
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # batch x 32 x 32 x 32 -> batch x 32 x 16 x 16
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # batch x 8 x 16 x 16 -> batch x 8 x 8 x 8
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            # batch x 4 x 8 x 8 -> batch x 4 x 4 x 4
            nn.MaxPool2d(2, 2),
        )

        # Fully Connected Layer
        self.fcLayer = nn.Sequential(
            nn.Linear(4 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        self.batch = x.shape[0]
        out = self.layer(x)
        out = out.view(self.batch, -1)
        out = self.fcLayer(out)
        return out
