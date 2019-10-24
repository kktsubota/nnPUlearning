import torch
from torch import nn


class MyClassifier(nn.Module):
    prior = 0.0

    def forward(self, x, t, loss_func):
        self.clear()
        h = self.calculate(x)
        self.loss = loss_func(h, t)
        return self.loss
    
    def clear(self):
        self.loss = None
    
    def calculate(self, x):
        return None
    
    def error(self, x, t):
        training = self.training
        self.eval()
        size = len(t)
        with torch.no_grad():
            h = torch.reshape(torch.sign(self.calculate(x)), (size,))
            result = (h != t).sum().float() / size
        self.train(training)
        return result.cpu()

    def compute_prediction_summary(self, x, t):
        n_p = (t == 1).sum()
        n_n = (t == -1).sum()
        with torch.no_grad():
            h = torch.flatten(torch.sign(self.calculate(x)))
        t_p = ((h == 1) * (t == 1)).sum()
        t_n = ((h == -1) * (t == -1)).sum()
        f_p = n_n - t_n
        f_n = n_p - t_p
        return t_p.item(), t_n.item(), f_p.item(), f_n.item()


class LinearClassifier(MyClassifier, nn.Module):
    def __init__(self, prior, dim):
        super(LinearClassifier, self).__init__()
        self.l = nn.Linear(dim, 1)
        self.prior = prior

    def calculate(self, x):
        x = x.view(x.shape[0], -1)
        h = self.l(x)
        return h


class ThreeLayerPerceptron(MyClassifier, nn.Module):
    def __init__(self, prior, dim):
        super(ThreeLayerPerceptron, self).__init__()
        self.l1 = nn.Linear(dim, 100)
        self.l2 = nn.Linear(100, 1)
        self.af = nn.ReLU()
        self.prior = prior
    
    def calculate(self, x):
        x = x.view(x.shape[0], -1)
        h = self.l1(x)
        h = self.af(h)
        h = self.l2(h)
        return h


class MultiLayerPerceptron(MyClassifier, nn.Module):
    def __init__(self, prior, dim):
        super(MultiLayerPerceptron, self).__init__()
        self.l1 = nn.Linear(dim, 300, bias=False)
        self.b1 = nn.BatchNorm1d(300)
        self.l2 = nn.Linear(300, 300, bias=False)
        self.b2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300, bias=False)
        self.b3 = nn.BatchNorm1d(300)
        self.l4 = nn.Linear(300, 300, bias=False)
        self.b4 = nn.BatchNorm1d(300)
        self.l5 = nn.Linear(300, 1)

        self.af = nn.ReLU()
        self.prior = prior

    def calculate(self, x):
        x = x.view(x.shape[0], -1)
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.l4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.l5(h)
        return h


class CNN(MyClassifier, nn.Module):
    def __init__(self, prior, dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(96)
        self.b3 = nn.BatchNorm2d(96)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(192)
        self.b6 = nn.BatchNorm2d(192)
        self.b7 = nn.BatchNorm2d(192)
        self.b8 = nn.BatchNorm2d(192)
        self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10*8*8, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

        self.af = nn.ReLU()
        self.prior = prior

    def calculate(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
