'''
Asterisk Vanadium Project
Set Hyper Parameters
'''

import torch
import random
from Models import LinearClassifier, CNN, FCLayer
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

batch_size = 128
epoch = 5


def train(loader):
    model.train()
    for idx, [image, label] in enumerate(loader):
        trainer.step(image, label)


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
        print("Final Result - accuracy: {}, lr: {}\n\n".format(100 * correct / 10000, lr))
        return correct
    else:
        print("epoch: {}, accuracy: {}".format(n_epoch, 100 * correct / 10000))


if __name__ == "__main__":
    train_load, valid_load, test_load = getCIFAR10.getCIFAR10(40000, batch_size)
    bestLR = -1.
    bestK = -1.
    maxCorrect = 0
    for cnt in range(100):
        lr = 10 ** (3 * random.random() - 5)
        k = 10 ** (10 * random.random() - 5)
        print("try: {}, lr: {}, k: {}".format(cnt+1, lr, k))
        model = CNN.CNN().to(device)
        trainer = Train_01.Trainer01(lr, model, device)

        for i in range(1, epoch + 1):
            trainer.lr = trainer.lr / k / i
            train(train_load)
            evaluate(valid_load, i)

        # Training Done
        with torch.no_grad():
            n_correct = evaluate(test_load, epoch + 1)
            if n_correct > maxCorrect:
                maxCorrect = n_correct
                bestLR = lr
                bestK = k

    print("best Accuracy: {}, learning rate: {}, bestK: {}".format(maxCorrect/100, bestLR, bestK))
