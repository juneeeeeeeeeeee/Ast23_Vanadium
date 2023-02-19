import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

batch_size = 128
epoch = 20

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

# Data Sets
trainset = datasets.CIFAR10(root='./', train=True, download=True)
testset = datasets.CIFAR10(root='./', train=False, download=True, transform=transform)


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.x = torch.tensor(data / 256., dtype=torch.float).permute(0, 3, 1, 2)
        self.x -= 0.5
        self.x /= 0.5
        self.y = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# Data Loaders
def Loaders(datas=trainset.data, labels=trainset.targets, idx_num=40000):
    train_datas = datas[:idx_num]
    valid_datas = datas[idx_num:]

    train_labels = labels[:idx_num]
    valid_labels = labels[idx_num:]

    train_ds = MyDataset(data=train_datas, label=train_labels)
    valid_ds = MyDataset(data=valid_datas, label=valid_labels)

    train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_load = DataLoader(valid_ds, batch_size=1, shuffle=False)

    print("Data Loading Completed")
    return train_load, valid_load


trainloader, validloader = Loaders()
testloader = DataLoader(testset, batch_size=1, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.batch = 0
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully Connected Layer
        self.fcLayer = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
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
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    print("CNN Test")
    print(device)
    model = CNN().to(device)
    trainer = Train(2e-3, model)

    # Training
    for i in range(epoch):
        trainer.lr = trainer.lr / (1 + i * 0.1)
        model.train()
        for image, label in trainloader:
            trainer.trainStep(image, label)
        model.eval()
        total = 0
        correct = 0
        for image, label in validloader:

            x = image.to(device)
            y = label.to(device)
            output = model.forward(x)
            outidx = torch.argmax(output)
            total += 1
            if outidx == y:
                correct += 1
        print("epoch: {}, accuracy: {}".format(i + 1, 100*correct/total))

    with torch.no_grad():
        total_test = 0
        correct_test = 0
        for image, label in testloader:
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x)
            outidx = torch.argmax(output)
            total_test += 1
            if outidx == y:
                correct_test += 1

        print("Accuracy: ", 100*correct_test/total_test)
        print(correct_test)
        print(total_test)

