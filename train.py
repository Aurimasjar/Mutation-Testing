import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from utils.model import Model

def my_loader(path):
    data = np.genfromtxt(path, delimiter=',')
    return data


def testAccuracy(model, test_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            samples, labels = data
            # run the model on the test set to predict labels
            outputs = model(samples)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Function to save the model
def saveModel(model):
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


# def train_model(vectors):
def train_model(batch_size, num_epochs):
    print('train model')
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5) todo think what normalization is needed
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    data = datasets.DatasetFolder(root='./data', loader=my_loader, extensions='.csv', transform=transform)
    print(data)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=1)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        data, [0.8, 0.1, 0.1]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    # print(train_dataset.__dict__)
    # print(valid_dataset.__dict__)
    # print(test_dataset.__dict__)

    model = Model(128, 300)
    # loss_fn = nn.CrossEntropyLoss() todo analyze did, how and why they used cross entropy for binary classification
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.eval()

    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (samples, labels) in enumerate(train_loader, 0):

            # for s1 in samples:
            #     for s2 in s1:
            #         for s3 in s2:
            #             for s4 in s3:
            #                 if s4.item() < 0 or s4.item() > 1:
            #                     print("FOUNDTHEGUILTY")
            #                     print('s4', s4.item())

            samples = samples.float()
            print('samples', samples)
            labels = labels.float()
            print('labels', labels)
            # samples = samples.unsqueeze(dim=0)
            # samples = samples.squeeze()
            print('samples after', samples.shape)
            # get the inputs
            samples = Variable(samples.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(samples)
            # compute the loss based on model output and real labels
            # loss = loss_fn(outputs, labels)
            loss = loss_fn(outputs, labels.view(len(labels), 1))
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 samples
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy(model, test_loader)
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % accuracy)

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy

    print('train model finished')
    return None
