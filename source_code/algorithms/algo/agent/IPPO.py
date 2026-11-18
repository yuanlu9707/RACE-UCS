from algorithms.algo.agent.DPPO import DPPOAgent
import torch.nn as nn
from algorithms.models import CategoricalActor, CategoricalActor0
from torch.optim import Adam
from algorithms.models import MLP

class IPPOAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        self.actors = self._init_actors()
        self.vs = self._init_vs()
        self._gru_hidden = {}
        self.last_gru_hidden_input = None
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)

    def _init_actors(self):
        actors = nn.ModuleList()
        plain_ncsu = (self.input_args.dataset == 'NCSU'
                      and not self.input_args.use_ritrl
                      and not self.input_args.use_cre)
        actor_class = CategoricalActor0 if plain_ncsu else CategoricalActor
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = self.observation_dim
            if self.input_args.obs_add_action:
                self.pi_args.sizes[0] += 2
            actors.append(actor_class(**self.pi_args._toDict()).to(self.device))
        return actors

    def _init_vs(self):
        vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.v_args.sizes[0] = self.observation_dim
            if self.input_args.obs_add_action:
                self.v_args.sizes[0] += 2
            if self.input_args.MAPPO:
                self.v_args.sizes[0] *= self.n_agent
            vs.append(MLP(**self.v_args._toDict()).to(self.device))
        return vs
