'''
Asterisk Vanadium Project
'''

import torch
import numpy as np

from Models import LinearClassifier, CNN, FCLayer, RESNet
from Trainers import Train_01
from GetDatas import getCIFAR10

# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu) # PoroSad
device = None
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

batch_size = 256
epoch = 20


def train(loader, n_epoch):
    sum_loss = 0
    model.train()
    for idx, [image, label] in enumerate(loader):
        sum_loss += trainer.step(image, label)
        if idx % 10 == 10 - 1:
            print("epoch: {}, loss: {}".format(n_epoch, sum_loss / 10))
            sum_loss = 0


def evaluate(loader, n_epoch):
    model.eval()
    correct = 0
    for image, label in loader:
        x = image.to(device)
        y = label.to(device)
        output = model.forward(x)
        result = torch.argmax(output, dim=1)
        correct += batch_size - torch.count_nonzero(result - y)
    print("epoch: {}, accuracy: {}\n\n".format(n_epoch, 100 * correct / 1000))


if __name__ == "__main__":
    print("main.py")
    print("Device on Working: ", device)

    model = RESNet.ResNet().to(device)
    trainer = Train_01.Trainer01(0.001, model, device)
    train_load, valid_load, test_load = getCIFAR10.getCIFAR10(49000, batch_size)

    model.load_state_dict(torch.load('model_params.pth'))

    for i in range(1, epoch + 1):
        train(train_load, i)
        evaluate(valid_load, i)

    # Training Done
    with torch.no_grad():
        model.eval()
        val = np.zeros(10, dtype=int)
        correct = 0
        for image, label in test_load:
            x = image.to(device)
            y = label.to(device)
            output = model.forward(x)
            result = torch.argmax(output, dim=1)
            for res, ans in zip(result, y):
                if res == ans:
                    val[res] += 1
            correct += batch_size - torch.count_nonzero(result - y)

        print("Final Result - accuracy: {}\n\n".format(100 * correct / 1000))
        for i in range(10):
            print("class {}: {} / 1000".format(i, val[i]))
        torch.save(model.state_dict(), 'model_params.pth')

