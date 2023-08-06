import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

np.random.seed(69)
torch.manual_seed(99)

class NeuralNet(nn.Module):
    def __init__(self, hidden_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out