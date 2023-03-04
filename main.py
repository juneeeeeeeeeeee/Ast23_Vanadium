'''
Asterisk Vanadium Project
'''

import torch

from Models import LinearClassifier, CNN
from Trainers import Train_01
from GetDatas import getCIFAR10

# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu)
device = None
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

batch_size = 256
epoch = 30


def train(loader, n_epoch):
    sum_loss = 0
    model.train()
    for idx, [image, label] in enumerate(loader):
        sum_loss += trainer.step(image, label)
        if idx % 15 == 14:
            print("epoch: {}, loss: {}".format(n_epoch, sum_loss / 15))
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
    if n_epoch > epoch:
        print("Final Result - accuracy: {}\n\n".format(100 * correct / 10000))
    else:
        print("epoch: {}, accuracy: {}\n\n".format(n_epoch, 100 * correct / 10000))


if __name__ == "__main__":
    print("main.py")
    print("Device on Working: ", device)

    model = CNN.CNN().to(device)
    trainer = Train_01.Trainer01(10e-4, model, device)
    train_load, valid_load, test_load = getCIFAR10.getCIFAR10(40000, batch_size)

    for i in range(1, epoch + 1):
        train(train_load, i)
        evaluate(valid_load, i)

    # Training Done
    with torch.no_grad():
        evaluate(test_load, epoch + 1)
