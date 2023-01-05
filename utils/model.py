from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()

        # Define the convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Define the max pooling layers
        self.pool = torch.nn.MaxPool2d(2, 2)
        # Define the fully-connected layers
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 32)
        self.fc6 = torch.nn.Linear(32, 16)
        self.fc7 = torch.nn.Linear(16, 8)
        self.fc8 = torch.nn.Linear(8, 4)
        self.fc9 = torch.nn.Linear(4, 2)
        self.fc10 = torch.nn.Linear(2, 1)

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        print('x_shape:', x.shape)
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = torch.nn.functional.relu(self.fc6(x))
        x = torch.nn.functional.relu(self.fc7(x))
        x = torch.nn.functional.relu(self.fc8(x))
        x = torch.nn.functional.relu(self.fc9(x))
        x = self.fc10(x)
        return x
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # return out
