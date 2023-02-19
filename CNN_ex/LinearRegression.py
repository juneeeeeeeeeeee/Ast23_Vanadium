import torch
import torch.nn as nn
# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu)
device = None
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class LR(nn.Module):
    # Linear Regression
    def __init__(self):
        super(LR, self).__init__()
