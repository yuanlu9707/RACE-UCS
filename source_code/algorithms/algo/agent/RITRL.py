"""RITRL adapter for the existing IPPO/MAPPO rollout and PPO interfaces."""
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from algorithms.algo.agent.IPPO import IPPOAgent
from algorithms.ritrl import RITRL
from algorithms.cre import CRE


class RITRLAgent(IPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        super().__init__(logger, device, agent_args, input_args)
        self.optimizer_aux = Adam([p for p in self.actors.parameters() if p.requires_grad],
                                  lr=input_args.ritrl_lr)
        self.use_cre = bool(getattr(input_args, 'use_cre', False))
        self._cre_history = None
        self._cre_returns = None
        self.last_cre_data = None
        if self.use_cre:
            self.cre = CRE(self.ritrl.hidden_dim,
                           self.ritrl.hidden_dim + input_args.ritrl_consensus_embed_dim,
                           self.pi_args.branchs, input_args).to(self.device)
            self.optimizer_cre = Adam(self.cre.main_parameters(), lr=input_args.cre_lr)
            self.optimizer_cre_score = Adam(self.cre.scorer_parameters(), lr=input_args.cre_score_lr)

    def _init_actors(self):
        # Previous actions have their own embedding, so do not reconstruct the
        # optional action columns appended by the legacy runner.
        return nn.ModuleList([RITRL(self.observation_dim, self.n_agent, self.pi_args.branchs,
                                   self.input_args).to(self.device)])

    @property
    def ritrl(self):
        return self.actors[0]

    def _initial_state(self, obs):
        h = obs.new_zeros(obs.size(0), self.n_agent, self.ritrl.hidden_dim)
        previous = obs.new_full((obs.size(0), self.n_agent, 2), -1)
        return torch.cat([h, h.clone(), previous], -1)

    def reset_gru_hidden(self, phase, reset_mask=None):
        if self.use_cre and phase == 'train' and self._cre_history is not None:
            if reset_mask is None:
                self._cre_history = None
                self._cre_returns = None
            else:
                self._reset_cre(torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool))
        if reset_mask is None:
            self._gru_hidden.pop(phase, None)
        elif phase in self._gru_hidden:
            state = self._gru_hidden[phase]
            mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool)
            state[mask] = self._initial_state(state)[mask]

    @torch.no_grad()
    def record_actions(self, actions, phase):
        self._gru_hidden[phase][..., -2:] = torch.as_tensor(actions, device=self.device)

    @torch.no_grad()
    def act(self, s, phase):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        state = self._gru_hidden.get(phase)
        if state is None or state.size(0) != s.size(0):
            state = self._initial_state(s)
        self.last_gru_hidden_input = state.detach().clone()
        h, target_h, previous = torch.split(state, [self.ritrl.hidden_dim, self.ritrl.hidden_dim, 2], -1)
        obs = s[..., :self.observation_dim]
        probs, h, _, context = self.ritrl.infer(obs, previous, h)
        if self.use_cre and phase == 'train':
            _, embedding = self.ritrl.consensus_features(h, context)
            self._cre_current = (h.detach(), torch.cat([context, embedding], -1).detach())
        if phase == 'train':
            target_h = self.ritrl.target_encoder(obs, previous, target_h)
        self._gru_hidden[phase] = torch.cat([h, target_h, previous], -1).detach()
        split = self.pi_args.branchs[0]
        return {'branch1': Categorical(probs[..., :split]),
                'branch2': Categorical(probs[..., split:])}

    def get_logp_with_hidden(self, s, a, hidden):
        h, _, previous = torch.split(hidden.detach(), [self.ritrl.hidden_dim, self.ritrl.hidden_dim, 2], -1)
        probs, _, _, _ = self.ritrl.infer(s.to(self.device)[..., :self.observation_dim], previous, h)
        split = self.pi_args.branchs[0]
        return torch.cat([probs[..., :split].gather(-1, a[..., :1].long()).clamp_min(1e-12).log(),
                          probs[..., split:].gather(-1, a[..., 1:2].long()).clamp_min(1e-12).log()], -1)

    def get_logp(self, s, a):
        raise RuntimeError('RITRL log probabilities require stored recurrent state')

    def _evalV(self, s):
        s = s.to(self.device)
        if self.input_args.MAPPO:
            global_obs = s.reshape(s.size(0), -1)
            return torch.stack([v(global_obs) for v in self.vs], 1)
        return torch.stack([v(s[:, i]) for i, v in enumerate(self.vs)], 1)

    @torch.no_grad()
    def _process_traj(self, **kwargs):
        # PPO advantages/returns are fixed targets for actor and critic updates.
        return super()._process_traj(**kwargs)

    def updateAgent(self, trajs, clip=None):
        count = int(self.ritrl.aux_updates.item())
        args = self.input_args
        stage = (0 if count < args.ritrl_rec_warmup else
                 1 if count < args.ritrl_rec_warmup + args.ritrl_cl_warmup else 2)
        # Optional reconstruction/contrastive warmup before policy optimization.
        if stage == 2 and self.use_cre:
            # Freeze h, m and P_on generation during PPO. W_c is a trainable
            # policy adapter, along with fusion, the direct observation path,
            # the concatenated policy MLP and action heads.
            # Remove gradients left by the previous auxiliary update so Adam
            # cannot apply momentum updates to newly frozen parameters.
            self.optimizer_pi.zero_grad(set_to_none=True)
            flags = [(p, p.requires_grad) for p in self.actors.parameters()]
            try:
                for p, _ in flags:
                    p.requires_grad_(False)
                self.ritrl.fusion.requires_grad_(True)
                self.ritrl.policy_obs.requires_grad_(True)
                self.ritrl.policy_mlp.requires_grad_(True)
                self.ritrl.heads.requires_grad_(True)
                self.ritrl.consensus_embedding.requires_grad_(True)
                result = super().updateAgent(trajs, clip)
            finally:
                for p, enabled in flags:
                    p.requires_grad_(enabled)
        else:
            result = super().updateAgent(trajs, clip) if stage == 2 else [0., 0., 0.]
        if self.use_cre:
            self._train_cre(trajs)
        obs = torch.stack([t['s'] for t in trajs]).to(self.device)
        done = torch.stack([t['d'] for t in trajs]).to(self.device)
        stored = torch.stack([t.gru_h for t in trajs]).to(self.device)
        visibility = (torch.stack([t.user_visibility for t in trajs]).to(self.device)
                      if all(hasattr(t, 'user_visibility') for t in trajs) else None)
        uav_visibility = (torch.stack([t.uav_visibility for t in trajs]).to(self.device)
                          if all(hasattr(t, 'uav_visibility') for t in trajs) else None)
        totals, chunks = {}, 0
        for _ in range(args.ritrl_aux_epochs):
            for start in range(0, obs.size(1), args.ritrl_sequence_length):
                h, target_h, _ = torch.split(stored[:, start].detach(),
                                            [self.ritrl.hidden_dim, self.ritrl.hidden_dim, 2], -1)
                losses, centers = [], []
                for t in range(start, min(start + args.ritrl_sequence_length, obs.size(1))):
                    previous = stored[:, t, :, -2:]
                    loss, h, target_h, center, metrics = self.ritrl.auxiliary_loss(
                        obs[:, t, :, :self.observation_dim], previous, h, target_h, stage,
                        visibility[:, t] if visibility is not None else None,
                        uav_visibility[:, t] if uav_visibility is not None else None)
                    losses.append(loss)
                    centers.append(center)
                    for name, value in metrics.items():
                        totals[name] = totals.get(name, 0.) + value.item()
                    chunks += 1
                    alive = (~done[:, t].bool()).to(h.dtype)
                    h, target_h = h * alive, target_h * alive
                self.optimizer_aux.zero_grad()
                torch.stack(losses).mean().backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), args.ritrl_grad_clip)
                self.optimizer_aux.step()
                self.ritrl.update_targets(torch.stack(centers).mean(0) if stage == 2 else None)
        self.ritrl.aux_updates.add_(1)
        self.logger.log(**{key: value / chunks for key, value in totals.items()}, ritrl_stage=stage)
        return result

    @torch.no_grad()
    def _reset_cre(self, mask):
        if self._cre_returns is not None:
            self._cre_returns[mask] = 0
        history = self._cre_history
        history[mask] = 0
        count = int(mask.sum().item())
        ids = torch.arange(count, device=self.device) + self.cre.next_episode_id
        history[mask, :, -1] = ids.to(history.dtype).unsqueeze(-1)
        self.cre.next_episode_id.add_(count)

    @torch.no_grad()
    def cre_transition(self, next_obs, action, external_reward, done):
        """Called only during training, after env.step; evaluation keeps task rewards."""
        import numpy as np
        h, raw = self._cre_current
        b, n, _ = h.shape
        c = self.cre.raw_context_dim
        if self._cre_history is None or self._cre_history.size(0) != b:
            self._cre_returns = None
            self._cre_history = h.new_zeros(b, n, c + 3)  # raw, reward, valid, episode ID
            self._reset_cre(torch.ones(b, device=self.device, dtype=torch.bool))
        history = self._cre_history
        actions = torch.as_tensor(action, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=h.dtype)[..., :self.observation_dim]
        target_h = self._gru_hidden['train'][..., self.ritrl.hidden_dim:2*self.ritrl.hidden_dim]
        next_h = self.ritrl.target_encoder(next_obs, actions, target_h)
        self.last_cre_data = torch.cat([h, raw, history[..., :c], next_h, history[..., c:]], -1).detach().clone()
        intrinsic = self.cre.intrinsic_reward(h, actions)
        if self._cre_returns is None or self._cre_returns.shape != intrinsic.shape:
            self._cre_returns = torch.zeros_like(intrinsic)
        self._cre_returns.mul_(self.input_args.cre_norm_discount).add_(intrinsic)
        self.cre.reward_normalizer.update(self._cre_returns)
        normalized = self.cre.reward_normalizer(intrinsic)
        terminal = torch.as_tensor(done, device=self.device, dtype=torch.bool).reshape(b, n)
        self._cre_returns.masked_fill_(terminal, 0)
        reward = torch.as_tensor(external_reward, device=self.device, dtype=h.dtype).reshape(b, n)
        augmented = reward + self.input_args.cre_reward_weight * normalized.mean(1, keepdim=True)
        history[..., :c] = raw
        history[..., c:c+1] = reward.mean(1, keepdim=True).unsqueeze(-1)
        history[..., c+1] = 1
        self.logger.log(cre_intrinsic_reward=intrinsic.mean(), cre_external_reward=reward.mean(),
                        cre_normalized_reward=normalized.mean(),
                        cre_reward_scale=self.cre.reward_normalizer.scale(), cre_training_reward=augmented.mean())
        return augmented.cpu().numpy().reshape(np.shape(external_reward))

    def _train_cre(self, trajs):
        if not all(hasattr(t, 'cre_data') for t in trajs):
            raise ValueError('CRE requires cre_transition data for each rollout step')
        data = torch.cat([t.cre_data for t in trajs], 0).to(self.device).flatten(0, 1).detach()
        actions = torch.cat([t['a'] for t in trajs], 0).to(self.device).flatten(0, 1)
        done = torch.cat([t['d'] for t in trajs], 0).to(self.device).flatten(0, 1)
        h, c = self.ritrl.hidden_dim, self.cre.raw_context_dim
        fields = torch.split(data, [h, c, c, h, 1, 1, 1], -1)
        hidden, raw, previous, next_h, reward, valid, ids = fields
        totals, batches = {}, 0
        for _ in range(self.input_args.cre_epochs):
            order = torch.randperm(data.size(0), device=self.device)
            for start in range(0, data.size(0), self.input_args.cre_batch_size):
                idx = order[start:start + self.input_args.cre_batch_size]
                # Independent positive/negative classification step. This loss
                # is not added to the three-term CRE calibration/prediction loss.
                self.optimizer_cre_score.zero_grad()
                score_loss, score_count = self.cre.scorer_loss(
                    raw[idx], previous[idx], reward[idx], ids[idx, 0].long(), valid[idx])
                if score_count.item() > 0:
                    score_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.cre.scorer_parameters()), self.input_args.cre_grad_clip)
                    self.optimizer_cre_score.step()
                self.optimizer_cre_score.zero_grad(set_to_none=True)
                loss, metrics = self.cre.loss(hidden[idx], raw[idx], previous[idx], next_h[idx],
                                              reward[idx], valid[idx], ids[idx, 0].long(), actions[idx], done[idx])
                self.optimizer_cre.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.cre.main_parameters(), self.input_args.cre_grad_clip)
                self.optimizer_cre.step()
                self.cre.update_target()
                metrics['cre_score_loss'] = score_loss.detach()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value.item() * idx.numel()
                batches += idx.numel()
        self.cre.enqueue(raw, ids[:, 0].long())
        self.logger.log(**{key: value / batches for key, value in totals.items()})

    def save_nets(self, dir_name, iter=0, is_newbest=False):
        super().save_nets(dir_name, iter, is_newbest)
        if self.use_cre:
            prefix = 'best' if is_newbest else str(iter)
            torch.save(self.cre.state_dict(), f'{dir_name}/Models/{prefix}_cre.pt')

    def load_nets(self, dir_name, iter=0, best=False):
        super().load_nets(dir_name, iter, best)
        if self.use_cre:
            prefix = 'best' if best else str(iter)
            state = torch.load(f'{dir_name}/Models/{prefix}_cre.pt', map_location=self.device)
            # Earlier CRE checkpoints have the same network shapes but no
            # reward-scale moments. Initialize only those newly added buffers.
            for key, value in self.cre.state_dict().items():
                if key.startswith('reward_normalizer.') and key not in state:
                    state[key] = value
            self.cre.load_state_dict(state)
            self._cre_history = None
            self._cre_returns = None
