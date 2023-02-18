import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

batch_size = 128
epoch = 100

# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu)
device = None
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

trainset = datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./', train=False, download=True, transform=transform)


class MyDataSet(Dataset):
    def __init__(self):
        pass


trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.batch = 0
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
           nn.MaxPool2d(2, 2),
        )

        # Fully Connected Layer
        self.fcLayer = nn.Sequential(
            nn.Linear(32 * 5 * 5, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, X):
        self.batch = X.shape[0]
        out = self.layer(X)
        out = out.view(self.batch, -1)
        out = self.fcLayer(out)
        return out


class Train:
    def __init__(self, lr, model):
        self.lr = lr
        self.model = model
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def trainStep(self, image, label):
        x = image.to(device)
        y = label.to(device)
        self.optimizer.zero_grad()
        output = model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    print("CNN Test")
    print(device)
    model = CNN()
    model.to(device)
    trainer = Train(1e-3, model)

    for i in range(epoch):
        for image, label in trainloader:
            trainer.trainStep(image, label)
        print("epoch:", i)

    total = 0
    correct = 0

    with torch.no_grad():
        for image, label in testloader:
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x)
            outidx = torch.argmax(output)
            total += 1
            if outidx == y:
                correct += 1

        print("Accuracy: ", 100*correct/total)
        print(correct)
        print(total)

