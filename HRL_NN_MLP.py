import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class FFNN_hlp(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(FFNN_hlp, self).__init__()
    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  def forward(self, obs):
      activation1 = F.relu(self.layer1(obs))
      activation2 = F.relu(self.layer2(activation1))
      output = self.layer3(activation2)
      return output


class FFNN_llp(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(FFNN_llp, self).__init__()
    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  def forward(self, obs):
      activation1 = F.relu(self.layer1(obs))
      activation2 = F.relu(self.layer2(activation1))
      output = self.layer3(activation2)
      return output


class high_level_policy(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(high_level_policy, self).__init__()
    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  def forward(self, obs):
      activation1 = F.relu(self.layer1(obs))
      activation2 = F.relu(self.layer2(activation1))
      output = self.layer3(activation2)
      return output


class low_level_policy(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(low_level_policy, self).__init__()
    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  def forward(self, obs):
      activation1 = F.relu(self.layer1(obs))
      activation2 = F.relu(self.layer2(activation1))
      output = self.layer3(activation2)
      return output


# implementing an embedding layer to map discrete actions to dense vectors
class ActionEmbedding(nn.Module):
    def __init__(self, num_actions, embedding_dim):
        super(ActionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)

    def forward(self, action):

        return self.embedding(action)
