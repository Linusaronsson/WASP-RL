import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, out_size, layer_size):
        super(MLP, self).__init__()

        self.fc_0 = nn.Linear(in_size, layer_size)
        self.fc_1 = nn.Linear(layer_size, layer_size)
        self.fc_2 = nn.Linear(layer_size, layer_size)

        self.fc_out = nn.Linear(layer_size, out_size)

    def forward(self, inp):
        x = torch.relu(self.fc_0(inp))
        x = torch.relu(self.fc_1(x))
        x = torch.relu(self.fc_2(x))

        x = self.fc_out(x)
        return x
