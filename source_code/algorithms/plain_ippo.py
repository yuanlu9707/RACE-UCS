"""JSAC3 CategoricalActor0, with optional arguments for the plain IPPO API.

Retains the reference module layout and initialization order, including idle
layers, so a fixed seed initializes successive UAV actors as in JSAC3.
MATCH/RITRL use their own actor branches. MLP is the common project builder.
"""
import torch
from torch import nn
from algorithms.models import MLP


class CategoricalActor0(nn.Module):
    """
    always returns a distribution
    """

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        net_fn = net_args['network']  # MLP
        self.snrmap_features = net_args.get('snrmap_features', 0)

        if self.snrmap_features > 0:
            # net_args['sizes'][0] -= self.snrmap_features
            # net_args['sizes'][-1] = 32
            # snr_net_args = copy.deepcopy(net_args)
            # snr_net_args['sizes'] = [32 + self.snrmap_features, 32, 9]

            # self.snr_network = net_fn(**snr_net_args)
            pass

        self.network = net_fn(**net_args)
        if net_args.get('comm', False):
            # self.branch1 = nn.Linear(64 + self.snrmap_features, net_args['branchs'][0])
            # self.branch2 = nn.Linear(64 + self.snrmap_features, net_args['branchs'][1])
            self.branch1 = nn.Linear(64, net_args['branchs'][0])
            self.branch2 = nn.Linear(64, net_args['branchs'][1])
            self.mlp2 = MLP([128, 64, 64])
            self.lin1 = nn.Linear(128, 64)
        else:
            self.branch1 = nn.Linear(net_args['sizes'][-1], net_args['branchs'][0])
            self.branch2 = nn.Linear(net_args['sizes'][-1], net_args['branchs'][1])
            # self.branch1 = nn.Linear(net_args['sizes'][-1] + self.snrmap_features, net_args['branchs'][0])
            # self.branch2 = nn.Linear(net_args['sizes'][-1] + self.snrmap_features, net_args['branchs'][1])
            self.mlp2 = MLP([64, 64, 64])
            self.lin1 = nn.Linear(64, 64)
        self.eps = 1e-5
        # if pi becomes truely deterministic (e.g. SAC alpha = 0)
        # q will become NaN, use eps to increase stability
        # and make SAC compatible with "Hard"ActorCritic

        # self.lstm = nn.Sequential(nn.LSTM(100, 64))
        # self.conv1 = nn.Conv1d(1, 1, 1, stride=1, padding=1, dilation=1)
        # self.lstm1 = nn.LSTM(100, 32)
        # self.lstm = nn.LSTMCell(100, 32)
        input_size = 64 + 48
        hidden_size = 32
        num_layers = 2
        num_classes = 100

        # self.lstm = BLSTM(input_size, hidden_size, num_layers, num_classes)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, obs, step=None, comm_mess=None):  # 多维度动作确认OK
        # obs [B,S]  #
        # if self.snrmap_features > 0:
        #     # x = self.network(obs[:, 0:-self.snrmap_features])
        #     # snrmap = obs[:, -self.snrmap_features:]
        #     # embed = self.snr_network(torch.cat([x, snrmap], dim=-1))
        # else:
        #     assert self.snrmap_features == 0
        #     embed = self.network(obs)

        if comm_mess is None:
            comm_mess = []
        embed = self.network(obs)
        if comm_mess != []:
            embed = torch.cat((embed, comm_mess), dim=1)
            embed = self.mlp2(embed)
        # embed = torch.relu(self.lin1(embed))
        # embed = self.mlp2(embed)

        # if self.snrmap_features > 0:
        #     snrmap = obs[:, -self.snrmap_features:]
        #     embed = torch.cat([embed, snrmap], dim=-1)  # 直接shortcut，concat到送入两个branch之前的embedding上

        # embed = torch.relu(self.lin1(embed)) # embed = nn.ReLU(self.lin1(embed))
        # embed = embed.unsqueeze(1)
        # embed, _ = self.lstm(embed)
        # embed = embed.squeeze(1)

        logit1, logit2 = self.branch1(embed), self.branch2(embed)
        # logits = torch.stack([logit1, logit2], dim=1)
        prob1 = self.softmax(logit1) + self.eps
        prob1 = prob1 / prob1.sum(dim=-1, keepdim=True)

        prob2 = self.softmax(logit2) + self.eps
        prob2 = prob2 / prob2.sum(dim=-1, keepdim=True)
        return torch.cat([prob1, prob2], dim=-1)
