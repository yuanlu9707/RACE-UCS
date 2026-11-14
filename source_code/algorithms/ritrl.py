"""RITRL: local recurrent inference with centralized auxiliary supervision.

Tensor axes: [batch, observer UAV, target UAV, feature]. No real teammate
observation enters the policy forward pass; it is used only by auxiliary_loss.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim))


class TemporalEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, branches):
        super().__init__()
        self.obs = mlp(obs_dim, hidden_dim, hidden_dim)
        # Index 0 is reserved for the episode-start action (represented by -1).
        self.actions = nn.ModuleList([nn.Embedding(n + 1, action_dim) for n in branches])
        self.gru = nn.GRUCell(hidden_dim + len(branches) * action_dim, hidden_dim)

    def forward(self, obs, previous_action, hidden):
        action = torch.cat([emb(previous_action[..., i].long() + 1)
                            for i, emb in enumerate(self.actions)], dim=-1)
        features = torch.cat([self.obs(obs), action], dim=-1)
        return self.gru(features.reshape(-1, features.size(-1)),
                        hidden.reshape(-1, hidden.size(-1))).reshape_as(hidden)


class RITRL(nn.Module):
    def __init__(self, obs_dim, n_agent, branches, args):
        super().__init__()
        if n_agent < 2:
            raise ValueError('RITRL requires at least two UAVs')
        self.args, self.n_agent, self.obs_dim = args, n_agent, obs_dim
        # EnvMobile layout: UAV xy; per-user [xy,AoI,future...]; map; time.
        frames = 4 if args.use_stack_frame else 1
        self.frame_dim = obs_dim // frames
        stride = 3 * (1 + args.future_obs)
        expected = 2 * n_agent + args.poi_num * stride + 1 + (args.map_size ** 2 if args.use_snrmap else 0)
        if obs_dim != expected * frames:
            raise ValueError('RITRL observation layout does not match UAV/user/frame configuration')
        indices = (torch.arange(args.poi_num)[:, None] * stride + torch.arange(3)[None, :]
                   + 2 * n_agent + (frames - 1) * self.frame_dim).reshape(-1)
        self.register_buffer('user_indices', indices, persistent=False)
        reconstruction_indices = torch.cat([torch.arange(2 * n_agent) + (frames - 1) * self.frame_dim, indices])
        self.register_buffer('reconstruction_indices', reconstruction_indices, persistent=False)
        self.hidden_dim = h = args.ritrl_hidden_dim
        self.encoder = TemporalEncoder(obs_dim, h, args.ritrl_action_embed_dim, branches)
        self.identity = nn.Embedding(n_agent, args.ritrl_id_dim)
        self.decoder = mlp(h + args.ritrl_id_dim, h, h)
        # Current UAV positions and user triples; no time/map/future/action outputs.
        self.observation_head = nn.Linear(h, 2 * n_agent + 3 * args.poi_num)
        self.projector = mlp(h, h, args.ritrl_projection_dim)  # g_rec
        self.query, self.key, self.value = [nn.Linear(h, h) for _ in range(3)]
        self.belief = mlp(2 * h, h, h)
        self.consensus = mlp(h, h, args.ritrl_consensus_classes)
        # Keep the checkpoint-compatible [K_c, embedding_dim] parameter layout;
        # P_on @ weight is W_c P_on in column-vector notation.
        self.consensus_embedding = nn.Embedding(args.ritrl_consensus_classes,
                                                 args.ritrl_consensus_embed_dim)
        self.fusion = mlp(2 * h + args.ritrl_consensus_embed_dim, h, h)
        # A direct local-observation path complements the recurrent team features.
        self.policy_obs = mlp(obs_dim, h, h)
        self.policy_mlp = mlp(2 * h, h, h)
        self.heads = nn.ModuleList([nn.Linear(h, n) for n in branches])
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)  # CRE future-state target only
        # g_tar is trainable; stop-gradient is applied BEFORE it, to h_t^v.
        # Preserve checkpoint keys while removing the previous EMA semantics.
        self.target_projector = deepcopy(self.projector)
        self.target_consensus = deepcopy(self.consensus).requires_grad_(False)
        self.register_buffer('center', torch.zeros(args.ritrl_consensus_classes))
        self.register_buffer('aux_updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('off_diagonal', ~torch.eye(n_agent, dtype=torch.bool))

    def consensus_features(self, hidden, context):
        """Shared continuous team representation for both Actor and CRE."""
        belief = self.belief(torch.cat([hidden, context], dim=-1))
        probabilities = F.softmax(self.consensus(belief) / self.args.ritrl_online_temperature, dim=-1)
        embedding = probabilities @ self.consensus_embedding.weight
        return probabilities, embedding

    def infer(self, obs, previous_action, hidden):
        h = self.encoder(obs, previous_action, hidden)
        b, n, width = h.shape
        identities = self.identity.weight.view(1, 1, n, -1).expand(b, n, -1, -1)
        source = h.unsqueeze(2).expand(-1, -1, n, -1)
        latent = self.decoder(torch.cat([source, identities], dim=-1))
        scores = (self.query(h).unsqueeze(2) * self.key(latent)).sum(-1) / math.sqrt(width)
        scores = scores.masked_fill(~self.off_diagonal, float('-inf'))
        context = (scores.softmax(-1).unsqueeze(-1) * self.value(latent)).sum(2)
        _, team_embedding = self.consensus_features(h, context)
        features = self.fusion(torch.cat([h, context, team_embedding], -1))
        policy_features = self.policy_mlp(torch.cat([self.policy_obs(obs), features], -1))
        probs = torch.cat([F.softmax(head(policy_features), -1) for head in self.heads], -1)
        return probs, h, latent, context

    def reconstruction_loss(self, prediction, obs, user_visibility=None, uav_visibility=None):
        target = obs.index_select(-1, self.user_indices).reshape(obs.size(0), self.n_agent, self.args.poi_num, 3)
        if user_visibility is None:
            # Convenience fallback for direct callers; rollouts supply the exact
            # environment mask, including visible users whose three values are 0.
            user_visibility = target.ne(0).any(-1)
        if user_visibility.shape != target.shape[:-1]:
            raise ValueError('user visibility must have shape [batch, UAV, user]')
        weights = torch.where(user_visibility.bool(), target.new_tensor(1.), target.new_tensor(self.args.ritrl_lambda0))
        uav_target = obs.index_select(-1, self.reconstruction_indices[:2*self.n_agent]).reshape(obs.size(0), self.n_agent, self.n_agent, 2)
        if uav_visibility is None:
            uav_visibility = uav_target.ne(0).any(-1) | torch.eye(self.n_agent, device=obs.device, dtype=torch.bool)
        if uav_visibility.shape != uav_target.shape[:-1]:
            raise ValueError('UAV visibility must have shape [batch, observer UAV, UAV slot]')
        uav_weights = torch.where(uav_visibility.bool(), target.new_tensor(1.), target.new_tensor(self.args.ritrl_lambda0))
        uav_prediction = prediction[..., :2*self.n_agent].reshape(obs.size(0), self.n_agent, self.n_agent, self.n_agent, 2)
        uav_squared = (uav_prediction - uav_target.unsqueeze(1)).square().sum(-1)
        uav_loss = (uav_squared * uav_weights.unsqueeze(1)).sum(-1) / uav_weights.sum(-1).unsqueeze(1)
        prediction = prediction[..., 2*self.n_agent:].reshape(obs.size(0), self.n_agent, self.n_agent, self.args.poi_num, 3)
        # Sum squared error over x,y,AoI exactly as in the supplied formula:
        # divide by sum_n w, NOT by 3*sum_n w and NOT by observation dimension.
        squared = (prediction - target.unsqueeze(1)).square().sum(-1)
        per_pair = (squared * weights.unsqueeze(1)).sum(-1) / weights.sum(-1).unsqueeze(1)
        return per_pair[:, self.off_diagonal].mean(), uav_loss[:, self.off_diagonal].mean()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Migrate the old full-observation head to the selected spatial rows.
        for name in ('weight', 'bias'):
            key = prefix + 'observation_head.' + name
            if key in state_dict and state_dict[key].size(0) == self.obs_dim:
                state_dict[key] = state_dict[key].index_select(0, self.reconstruction_indices.to(state_dict[key].device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def individual_contrastive_loss(self, latent, hidden):
        query = F.normalize(self.projector(latent), dim=-1)
        keys = F.normalize(self.target_projector(hidden.detach()), dim=-1)
        similarities = torch.einsum('buvd,bjd->buvj', query, keys) / self.args.ritrl_cl_temperature
        labels = torch.arange(self.n_agent, device=hidden.device).view(1, 1, -1)
        labels = labels.expand(hidden.size(0), self.n_agent, -1)[:, self.off_diagonal]
        # CE = NEGATIVE log positive probability. Both projectors learn; only
        # the reconstructed/query branch can send this gradient into the GRU.
        return F.cross_entropy(similarities[:, self.off_diagonal].reshape(-1, self.n_agent),
                               labels.reshape(-1))

    def auxiliary_loss(self, obs, previous_action, hidden, target_hidden, stage, user_visibility=None, uav_visibility=None):
        _, h, latent, context = self.infer(obs, previous_action, hidden)
        with torch.no_grad():
            target_h = self.target_encoder(obs, previous_action, target_hidden)
        prediction = self.observation_head(latent)
        user_rec, uav_rec = self.reconstruction_loss(prediction, obs, user_visibility, uav_visibility)
        rec = user_rec + uav_rec
        cl = self.individual_contrastive_loss(latent, h)
        # Consensus cannot backpropagate into encoder/decoder/attention.
        belief = self.belief(torch.cat([h.detach(), context.detach()], -1))
        online_logp = F.log_softmax(self.consensus(belief) / self.args.ritrl_online_temperature, -1)
        with torch.no_grad():
            target_logits = self.target_consensus(belief.detach())
            target_prob = F.softmax((target_logits - self.center) / self.args.ritrl_target_temperature, -1)
        cross_entropy = -torch.einsum('buk,bvk->buv', online_logp, target_prob)
        cons = cross_entropy[:, self.off_diagonal].mean()
        loss = self.args.ritrl_rec_weight * rec
        if stage >= 1:
            loss = loss + self.args.ritrl_cl_weight * cl
        if stage >= 2:
            loss = loss + self.args.ritrl_cons_weight * cons
        metrics = {'ritrl_rec_loss': rec.detach(), 'ritrl_cl_loss': cl.detach(),
                   'ritrl_ind_loss': cl.detach(),
                   'ritrl_user_rec_loss': user_rec.detach(), 'ritrl_uav_rec_loss': uav_rec.detach(),
                   'ritrl_cons_loss': cons.detach(), 'ritrl_aux_loss': loss.detach()}
        return loss, h, target_h, target_logits.detach().mean((0, 1)), metrics

    @torch.no_grad()
    def update_targets(self, mean_logits=None):
        beta = self.args.ritrl_ema
        for online, target in [(self.encoder, self.target_encoder),
                               (self.consensus, self.target_consensus)]:
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.mul_(beta).add_(parameter, alpha=1 - beta)
        if mean_logits is not None:
            momentum = self.args.ritrl_center_momentum
            self.center.mul_(momentum).add_(mean_logits, alpha=1 - momentum)
