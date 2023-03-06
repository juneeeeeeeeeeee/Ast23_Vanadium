import torch.nn as nn
import torch.nn.functional as F

class LCModel(nn.Module):
    # Linear Classifier
    def __init__(self):
        super(LCModel, self).__init__()
        self.batch = 0
        self.linear1 = nn.Linear(3072, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        self.batch = x.shape[0]
        x = x.view(self.batch, -1)
        x = self.linear1(x)
        # x = F.leaky_relu(x)
        x = self.linear2(x)
        # x = F.leaky_relu(x)
        x = self.linear3(x)
        return x
