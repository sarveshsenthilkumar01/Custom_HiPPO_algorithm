import torch as th
from click.core import F
from torch import nn
import torch.nn.functional as f
import numpy as np

class high_level_policy(nn.Module):
    def __init__(self):
        super(HierarchicalNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if (isinstance(obs, np.ndarray)):
            obs = th.tensor(obs, dtype = th.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output


class low_level_policy(nn.Module):
    def __init__(self):
        super(HierarchicalNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if (isinstance(obs, np.ndarray)):
            obs = th.tensor(obs, dtype=th.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

class FFNN(nn.Module):
    def __init__(self):
        super(HierarchicalNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if (isinstance(obs, np.ndarray)):
            obs = th.tensor(obs, dtype = th.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output
