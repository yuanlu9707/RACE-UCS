from algorithms.algo.agent.DPPO import DPPOAgent
import torch
import torch.nn as nn
from torch.distributions import Categorical
from algorithms.models import CategoricalActor, GRUCategoricalActor
from torch.optim import Adam
from algorithms.models import MLP

class IPPOAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        self.use_gru = bool(input_args.ues_GRU)
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        self.actors = self._init_actors()
        self.vs = self._init_vs()
        self._gru_hidden = {}
        self.last_gru_hidden_input = None
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)

    def _init_actors(self):
        actors = nn.ModuleList()
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = self.observation_dim
            if self.input_args.obs_add_action:
                self.pi_args.sizes[0] += 2
            actor_class = GRUCategoricalActor if self.use_gru else CategoricalActor
            actors.append(actor_class(**self.pi_args._toDict()).to(self.device))
        return actors

    def reset_gru_hidden(self, phase, reset_mask=None):
        if not self.use_gru:
            return
        if reset_mask is None:
            self._gru_hidden.pop(phase, None)
            return
        hidden = self._gru_hidden.get(phase)
        if hidden is not None:
            mask = torch.as_tensor(reset_mask, dtype=torch.bool, device=hidden.device)
            if mask.numel() != hidden.size(0):
                raise ValueError('GRU reset mask must contain one value per environment thread')
            self._gru_hidden[phase] = hidden.masked_fill(mask.view(-1, 1, 1), 0.0)

    def act(self, s, phase):
        if not self.use_gru:
            return super().act(s, phase)
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            if s.dim() != 3:
                raise ValueError('GRU actor expects observations [batch, n_agent, obs_dim]')
            s = self.get_networked_s(s, which_net='pi', phase=phase)
            hidden = self._gru_hidden.get(phase)
            hidden_dim = self.actors[0].hidden_dim
            if hidden is None or hidden.shape != (s.size(0), self.n_agent, hidden_dim):
                hidden = s.new_zeros(s.size(0), self.n_agent, hidden_dim)
            self.last_gru_hidden_input = hidden.detach().clone()
            outputs = [self.actors[i](s[:, i, :], hidden[:, i, :])
                       for i in range(self.n_agent)]
            probs = torch.stack([output[0] for output in outputs], dim=1)
            self._gru_hidden[phase] = torch.stack(
                [output[1] for output in outputs], dim=1
            ).detach()
            split = self.pi_args.branchs[0]
            return {
                'branch1': Categorical(probs[:, :, :split]),
                'branch2': Categorical(probs[:, :, split:]),
            }

    def get_logp_with_hidden(self, s, a, hidden):
        """Train each transition using its detached rollout-time GRU state."""
        if not self.use_gru:
            raise RuntimeError('get_logp_with_hidden is only used by the GRU actor')
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        hidden = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).detach()
        if s.dim() != 3 or hidden.shape != (s.size(0), self.n_agent, self.actors[0].hidden_dim):
            raise ValueError('GRU update expects states and hidden states [batch, n_agent, dim]')
        s = self.get_networked_s(s, which_net='pi', phase=None)
        split = self.pi_args.branchs[0]
        logps = []
        for i in range(self.n_agent):
            probs, _ = self.actors[i](s[:, i, :], hidden[:, i, :])
            logp1 = probs[:, :split].gather(-1, a[:, i, 0].long().unsqueeze(-1)).log()
            logp2 = probs[:, split:].gather(-1, a[:, i, 1].long().unsqueeze(-1)).log()
            logps.append(torch.cat([logp1, logp2], dim=-1))
        return torch.stack(logps, dim=1)

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
