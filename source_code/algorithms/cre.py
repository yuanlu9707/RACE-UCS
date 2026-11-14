"""Collaborative Representation-Calibrated Exploration (CRE).

The association scorer is optimized separately with -E[log gamma].
The main CRE objective is prediction NLL + alpha_I * (UB + LB).
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from algorithms.ritrl import mlp


class ReturnNormalizer(nn.Module):
    """Shared running standard deviation of discounted intrinsic returns.

    One unit-variance pseudo-sample stabilizes startup. Rewards are only scaled,
    never centered. Float64 parallel moments avoid cancellation in long runs.
    """
    def __init__(self, min_std):
        super().__init__()
        self.min_std = min_std
        self.register_buffer('count', torch.ones((), dtype=torch.float64))
        self.register_buffer('mean', torch.zeros((), dtype=torch.float64))
        self.register_buffer('m2', torch.ones((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, returns):
        values = returns.detach().double().reshape(-1)
        size = values.numel()
        delta = values.mean() - self.mean
        total = self.count + size
        self.m2.add_((values - values.mean()).square().sum() + delta.square() * self.count * size / total)
        self.mean.add_(delta * size / total)
        self.count.copy_(total)

    def scale(self):
        return (self.m2 / self.count).clamp_min(self.min_std ** 2).sqrt()

    def forward(self, reward):
        return reward / self.scale().to(reward.dtype)


class CRE(nn.Module):
    def __init__(self, hidden_dim, raw_context_dim, branches, args):
        super().__init__()
        self.args = args
        self.raw_context_dim = raw_context_dim
        width, state, latent = args.cre_hidden_dim, args.cre_state_dim, args.cre_latent_dim
        self.state = mlp(hidden_dim, width, state)
        self.target_state = deepcopy(self.state).requires_grad_(False)
        self.context = mlp(raw_context_dim, width, args.cre_context_dim)
        self.actions = nn.ModuleList([nn.Embedding(n, args.cre_action_embed_dim) for n in branches])
        self.posterior = mlp(state + len(branches) * args.cre_action_embed_dim, width, 2 * latent)
        self.predictor = mlp(latent, width, state)
        self.history_score = mlp(args.cre_context_dim + 1, width, args.cre_context_dim)
        self.current_score = mlp(args.cre_context_dim, width, args.cre_context_dim)
        self.calibration = mlp(args.cre_context_dim + 1, width, 1)
        self.reward_normalizer = ReturnNormalizer(args.cre_norm_min_std)
        self.register_buffer('memory', torch.zeros(args.cre_memory_size, raw_context_dim))
        self.register_buffer('memory_ids', torch.full((args.cre_memory_size,), -1, dtype=torch.long))
        self.register_buffer('memory_pointer', torch.zeros((), dtype=torch.long))
        self.register_buffer('next_episode_id', torch.zeros((), dtype=torch.long))

    def distribution(self, h, action):
        embedded = torch.cat([emb(action[..., i].long()) for i, emb in enumerate(self.actions)], -1)
        mu, logvar = self.posterior(torch.cat([self.state(h.detach()), embedded], -1)).chunk(2, -1)
        return mu, logvar.clamp(-6., 2.)

    @staticmethod
    def moments(mu, logvar):
        variance = logvar.exp()
        mean_psi = 0.5 * (mu.square() + variance - 1 - logvar).sum(-1)
        var_psi = (mu.square() * variance + 0.5 * (variance - 1).square()).sum(-1)
        return mean_psi.clamp_min(0), var_psi

    @torch.no_grad()
    def intrinsic_reward(self, h, action):
        return self.moments(*self.distribution(h, action))[0].sqrt()

    def scorer_parameters(self):
        for module in (self.history_score, self.current_score):
            yield from module.parameters()

    def main_parameters(self):
        scorer_ids = {id(p) for p in self.scorer_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in scorer_ids]

    def _association(self, raw, previous_raw, previous_reward, episode_ids, valid):
        """Stable log probability of the positive among independent candidates.

        Negative contexts come from DIFFERENT episodes only, never other UAVs
        or times in the same episode. Queue stores detached raw RITRL features.
        With no independent negatives/history the estimate is neutral (zero).
        """
        context = self.context(raw.detach())
        previous = self.context(previous_raw.detach())
        # The independent scorer step updates only g_theta, not f_col/RITRL.
        history = F.normalize(self.history_score(torch.cat([previous.detach(), previous_reward.detach()], -1)), dim=-1)
        positive_key = F.normalize(self.current_score(context.detach()), dim=-1)
        positive = (history * positive_key).sum(-1) / self.args.cre_temperature
        used = self.memory_ids >= 0
        pool_raw = torch.cat([raw.detach(), self.memory[used]], 0)
        pool_ids = torch.cat([episode_ids, self.memory_ids[used]], 0)
        eligible = pool_ids.unsqueeze(0) != episode_ids.unsqueeze(1)
        count = eligible.sum(-1).clamp_max(self.args.cre_negatives)
        # Random top-k among eligible entries avoids quadratic feature expansion.
        k = min(self.args.cre_negatives, pool_raw.size(0))
        random = torch.rand(eligible.shape, device=raw.device).masked_fill(~eligible, -1.)
        indices = random.topk(k, dim=-1).indices
        keys = F.normalize(self.current_score(self.context(pool_raw).detach()), dim=-1)
        negative = torch.einsum('bd,bkd->bk', history, keys[indices]) / self.args.cre_temperature
        negative = negative.masked_fill(~eligible.gather(1, indices), float('-inf'))
        log_probability = positive - torch.logsumexp(
            torch.cat([positive.unsqueeze(-1), negative], -1), -1)
        active = valid.reshape(-1).bool() & (count > 0)
        return log_probability, context, active

    def association(self, raw, previous_raw, previous_reward, episode_ids, valid):
        log_probability, context, active = self._association(raw, previous_raw, previous_reward, episode_ids, valid)
        return log_probability.exp() * active, context, active

    def scorer_loss(self, raw, previous_raw, previous_reward, episode_ids, valid):
        log_probability, _, active = self._association(raw, previous_raw, previous_reward, episode_ids, valid)
        loss = -(log_probability * active).sum() / active.sum().clamp_min(1)
        return loss, active.sum()

    def calibrated_mean(self, mean, variance, context, gamma):
        std = (variance + self.args.cre_epsilon).sqrt()
        direction = torch.tanh(self.calibration(torch.cat([context, mean.unsqueeze(-1)], -1)).squeeze(-1))
        # gamma controls amplitude outside H/tanh; the scorer has its own update.
        return mean + self.args.cre_rho * gamma.detach() * std * direction

    def loss(self, h, raw, previous_raw, next_h, previous_reward, valid, episode_ids, action, done):
        mu, logvar = self.distribution(h, action)
        mean, variance = self.moments(mu, logvar)
        sample = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        with torch.no_grad():
            target = self.target_state(next_h.detach())
        # Unit diagonal Gaussian negative log likelihood; mask terminal/reset transitions.
        nll = 0.5 * ((self.predictor(sample) - target).square() + math.log(2 * math.pi)).sum(-1)
        alive = (~done.reshape(-1).bool()).to(nll.dtype)
        pred = (nll * alive).sum() / alive.sum().clamp_min(1)
        gamma, context, active = self.association(raw, previous_raw, previous_reward, episode_ids, valid)
        std = (variance + self.args.cre_epsilon).sqrt()
        calibrated = self.calibrated_mean(mean, variance, context, gamma)
        beta = self.args.cre_rho + math.sqrt((1 - self.args.cre_tail_probability) / self.args.cre_tail_probability)
        upper = F.relu(calibrated + beta * std - self.args.cre_cmax).mean()
        lower = F.relu(self.args.cre_cmin - calibrated + beta * std).mean()
        loss = pred + self.args.cre_alpha_I * (upper + lower)
        return loss, {'cre_loss': loss.detach(), 'cre_pred_loss': pred.detach(),
                      'cre_upper_loss': upper.detach(), 'cre_lower_loss': lower.detach(),
                      'cre_gamma': gamma.detach().mean(), 'cre_kl': mean.detach().mean(),
                      'cre_association_active': active.float().mean()}

    @torch.no_grad()
    def update_target(self):
        for online, target in zip(self.state.parameters(), self.target_state.parameters()):
            target.mul_(self.args.cre_ema).add_(online, alpha=1 - self.args.cre_ema)

    @torch.no_grad()
    def enqueue(self, raw, episode_ids):
        for start in range(0, raw.size(0), self.memory.size(0)):
            chunk = raw[start:start + self.memory.size(0)]
            indices = (torch.arange(chunk.size(0), device=raw.device) + self.memory_pointer) % self.memory.size(0)
            self.memory[indices] = chunk.detach()
            self.memory_ids[indices] = episode_ids[start:start + chunk.size(0)]
            self.memory_pointer.copy_((self.memory_pointer + chunk.size(0)) % self.memory.size(0))
