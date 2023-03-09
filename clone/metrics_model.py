import torch.nn as nn
import torch


class MetricsModel(nn.Module):
    def __init__(self, metrics_dim, batch_size, means, stds):
        super(MetricsModel, self).__init__()
        self.metrics_dim = metrics_dim
        self.gpu = False
        self.batch_size = batch_size
        self.means = means
        self.stds = stds
        self.l1 = nn.Linear(self.metrics_dim, 200)
        self.l2 = nn.Linear(200, 200)
        self.l3 = nn.Linear(200, 200)
        self.l4 = nn.Linear(200, 200)
        self.l5 = nn.Linear(200, 200)
        self.l6 = nn.Linear(200, 100)
        self.l7 = nn.Linear(100, 50)
        self.l8 = nn.Linear(50, 25)
        self.l9 = nn.Linear(25, 1)


    def encode(self, x):
        x = torch.tensor(x).float()
        normalized_x = ((x - torch.tensor(self.means)) / torch.tensor(self.stds)).float()
        # normalized_x = ((x - torch.tensor(self.means) + 0.5) / (2 * torch.tensor(self.stds))).float()
        # TODO alternative: use torch.nn.functional.normalize()
        return normalized_x

    def forward(self, x1, x2):
        x1, x2 = self.encode(x1), self.encode(x2)
        x1, x2 = self.l1(x1), self.l1(x2)
        x1, x2 = self.l2(x1), self.l2(x2)
        x1, x2 = self.l3(x1), self.l3(x2)
        x1, x2 = self.l4(x1), self.l4(x2)
        x1, x2 = self.l5(x1), self.l5(x2)
        x = torch.abs(torch.add(x1, -x2))
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        y = torch.sigmoid(x)
        # y = torch.softmax(x)
        return y
