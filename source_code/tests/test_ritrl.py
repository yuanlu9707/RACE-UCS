import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gym.spaces import Discrete
from get_args import parse_args
from algorithms.ritrl import RITRL, IndependentRITRL
from algorithms.algo.agent.RITRL import RITRLAgent
from algorithms.algo.buffer import TrajectoryBuffer
from algorithms.config.Mobile_IPPO import getArgs


def arguments(*extra):
    # Isolate RITRL in these fixtures; RACE enables CRE by default in production.
    with patch.object(sys, 'argv', ['test', '--device', 'cpu', '--poi_num', '2', '--uav_num', '3', '--use_cre', 'False', *extra]):
        return parse_args()[0]


class Logger:
    def log(self, **kwargs):
        pass


def agent(args):
    env = SimpleNamespace(UAV_NUM=3, observation_space={'Box': SimpleNamespace(shape=(3, 13))},
                          action_space=[Discrete(9), Discrete(4)])
    config = getArgs(1, 1, env, args).agent_args
    config.pi_args.snrmap_features = 0
    config.n_update_pi = 2
    return RITRLAgent(Logger(), 'cpu', config, args)


class TestRITRL(unittest.TestCase):
    def test_independent_uav_networks_and_policy_gradients(self):
        args = arguments('--ritrl_hidden_dim', '16', '--ritrl_fusion_dim', '7')
        model = IndependentRITRL(13, 3, [9, 4], args)
        # Identical local inputs must still be routed to different parameters.
        obs = torch.rand(2, 1, 13).expand(-1, 3, -1)
        previous = torch.full((2, 3, 2), -1)
        hidden = torch.zeros(2, 3, 16)
        probs, _, _, _ = model.infer(obs, previous, hidden)
        self.assertEqual(probs.shape, (2, 3, 13))
        self.assertFalse(torch.allclose(probs[:, 0], probs[:, 1]))
        for component in (model.encoder, model.decoder, model.fusion, model.policy_obs):
            owners = [{id(p) for p in net.parameters()} for net in component.networks]
            self.assertFalse(owners[0] & owners[1])
        (-probs[:, 0, 0].log().mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.encoder.networks[0].parameters()))
        for u in (1, 2):
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                                for p in model.encoder.networks[u].parameters()))

    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.args = arguments('--ritrl_hidden_dim', '16', '--ritrl_sequence_length', '2')
        self.model = RITRL(13, 3, [9, 4], self.args)
        self.obs = torch.rand(2, 3, 13)
        self.obs[..., 6:9] = 0
        self.hidden = torch.zeros(2, 3, 16)
        self.previous = torch.full((2, 3, 2), -1)

    def test_weighted_user_and_uav_losses_match_formula(self):
        with torch.no_grad():
            self.model.observation_head.weight.zero_()
            self.model.observation_head.bias.fill_(1.)
        users = torch.tensor([[[True, False], [False, True], [False, False]]]).expand(2, -1, -1)
        uavs = torch.eye(3, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
        _, _, _, _, metrics = self.model.auxiliary_loss(
            self.obs, self.previous, self.hidden, self.hidden, 2, users, uavs)
        totals = []
        for mask, target, width in [(users, self.obs[..., 6:12], 3), (uavs, self.obs[..., :6], 2)]:
            expected = []
            for b in range(2):
                for u in range(3):
                    for v in range(3):
                        if u == v:
                            continue
                        weight = mask[b, v].float() * (1 - self.args.ritrl_lambda0) + self.args.ritrl_lambda0
                        squared = (1 - target[b, v].reshape(-1, width)).square().sum(-1)
                        expected.append((weight * squared).sum() / weight.sum())
            totals.append(torch.stack(expected).mean())
        torch.testing.assert_close(metrics['ritrl_user_rec_loss'], totals[0])
        torch.testing.assert_close(metrics['ritrl_uav_rec_loss'], totals[1])
        torch.testing.assert_close(metrics['ritrl_rec_loss'], sum(totals))

    def test_time_has_no_reconstruction_output_or_gradient(self):
        obs = self.obs.clone().requires_grad_(True)
        prediction = torch.zeros(2, 3, 3, 12, requires_grad=True)
        user_loss, uav_loss = self.model.reconstruction_loss(prediction, obs)
        (user_loss + uav_loss).backward()
        self.assertFalse(obs.grad[..., -1].any())
        changed = self.obs.clone()
        changed[..., -1] += 999.
        for a, b in zip(self.model.reconstruction_loss(prediction, self.obs), self.model.reconstruction_loss(prediction, changed)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(self.model.observation_head.out_features, 12)

    def test_lambda0_changes_relative_zero_slot_weight(self):
        obs = torch.zeros(1, 3, 13)
        obs[..., 6:9] = 1.
        prediction = torch.zeros(1, 3, 3, 12)
        visibility = torch.tensor([[[True, False]]]).expand(1, 3, -1)
        self.args.ritrl_lambda0 = 0.1
        small = self.model.reconstruction_loss(prediction, obs, visibility)[0]
        self.args.ritrl_lambda0 = 0.9
        large = self.model.reconstruction_loss(prediction, obs, visibility)[0]
        self.assertGreater(small, large)
        # No visible users: positive lambda0 keeps denominator finite.
        all_missing = self.model.reconstruction_loss(prediction, torch.zeros_like(obs), torch.zeros_like(visibility))[0]
        self.assertTrue(torch.isfinite(all_missing))

    def test_optional_features_and_stack_select_only_current_spatial_features(self):
        args = arguments('--future_obs', '1', '--use_snrmap', '--use-stack-frame')
        frame = 6 + 2 * 6 + args.map_size ** 2 + 1
        model = RITRL(frame * 4, 3, [9, 4], args)
        obs = torch.zeros(1, 3, frame * 4)
        pred = torch.ones(1, 3, 3, 12)
        baseline = model.reconstruction_loss(pred, obs)
        # All historical frames, future predictions, map, and time are excluded.
        excluded = torch.ones(frame * 4, dtype=torch.bool)
        excluded[model.reconstruction_indices] = False
        obs[..., excluded] = 100
        for a, b in zip(baseline, model.reconstruction_loss(pred, obs)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_old_observation_head_migration(self):
        state = self.model.state_dict()
        old_weight = torch.rand(13, 16)
        old_bias = torch.rand(13)
        state['observation_head.weight'] = old_weight
        state['observation_head.bias'] = old_bias
        self.model.load_state_dict(state)
        torch.testing.assert_close(self.model.observation_head.weight, old_weight[:12])
        torch.testing.assert_close(self.model.observation_head.bias, old_bias[:12])

    def test_policy_has_no_teammate_observation_leak(self):
        first = self.model.infer(self.obs, self.previous, self.hidden)[0]
        changed = self.obs.clone()
        changed[:, 1:] += 100
        second = self.model.infer(changed, self.previous, self.hidden)[0]
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)

    def test_policy_concatenates_direct_observation_and_fused_features(self):
        captured = []
        hook = self.model.policy_mlp.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            probs, h, _, context = self.model.infer(self.obs, self.previous, self.hidden)
        finally:
            hook.remove()
        _, embedding = self.model.consensus_features(h, context)
        fused = self.model.fusion(torch.cat([h, context, embedding], -1))
        expected_input = torch.cat([self.model.policy_obs(self.obs), fused], -1)
        torch.testing.assert_close(captured[0], expected_input)
        features = self.model.policy_mlp(expected_input)
        expected_probs = torch.cat([head(features).softmax(-1) for head in self.model.heads], -1)
        torch.testing.assert_close(probs, expected_probs)
        for branch in (probs[..., :9], probs[..., 9:]):
            torch.testing.assert_close(branch.sum(-1), torch.ones(2, 3))
        (-probs[..., 0].log().mean() - probs[..., 9].log().mean()).backward()
        for module in (self.model.policy_obs, self.model.fusion, self.model.policy_mlp):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

        # Even with the recurrent/team fusion held fixed, the policy can react
        # to its own current observation through the new direct path.
        with patch.object(self.model.fusion, 'forward', return_value=torch.zeros_like(fused)):
            first = self.model.infer(self.obs, self.previous, self.hidden)[0]
            changed = self.obs.clone()
            changed[:, 0] += 1.
            second = self.model.infer(changed, self.previous, self.hidden)[0]
        self.assertFalse(torch.equal(first[:, 0], second[:, 0]))
        torch.testing.assert_close(first[:, 1:], second[:, 1:], rtol=0, atol=0)

    def test_contrastive_and_consensus_formulas(self):
        from torch.nn import functional as F
        _, h, latent, context = self.model.infer(self.obs, self.previous, self.hidden)
        _, _, _, _, metrics = self.model.auxiliary_loss(
            self.obs, self.previous, self.hidden, self.hidden, 2)
        with torch.no_grad():
            keys = F.normalize(self.model.target_projector(h.detach()), dim=-1)
            query = F.normalize(self.model.projector(latent), dim=-1)
            logits = torch.einsum('buvd,bjd->buvj', query, keys) / self.args.ritrl_cl_temperature
            cl = torch.stack([-logits[:, u, v].log_softmax(-1)[:, v].mean()
                              for u in range(3) for v in range(3) if u != v]).mean()
            b = self.model.belief(torch.cat([h, context], -1))
            on = (self.model.consensus(b) / self.args.ritrl_online_temperature).log_softmax(-1)
            target = ((self.model.target_consensus(b) - self.model.center) /
                      self.args.ritrl_target_temperature).softmax(-1)
            cons = torch.stack([-(target[:, v] * on[:, u]).sum(-1).mean()
                                for u in range(3) for v in range(3) if u != v]).mean()
        torch.testing.assert_close(metrics['ritrl_cl_loss'], cl)
        torch.testing.assert_close(metrics['ritrl_ind_loss'], cl)
        torch.testing.assert_close(metrics['ritrl_cons_loss'], cons)

    def test_continuous_consensus_formula_and_actor_features(self):
        captured = []
        hook = self.model.fusion.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            _, hidden, _, context = self.model.infer(self.obs, self.previous, self.hidden)
        finally:
            hook.remove()
        probability, embedding = self.model.consensus_features(hidden, context)
        belief = self.model.belief(torch.cat([hidden, context], -1))
        expected_probability = (self.model.consensus(belief) / self.args.ritrl_online_temperature).softmax(-1)
        expected_embedding = sum(expected_probability[..., k:k+1] * self.model.consensus_embedding.weight[k]
                                 for k in range(self.args.ritrl_consensus_classes))
        torch.testing.assert_close(probability, expected_probability)
        torch.testing.assert_close(embedding, expected_embedding)
        torch.testing.assert_close(captured[0][..., -self.args.ritrl_consensus_embed_dim:], expected_embedding)

    def test_continuous_embedding_uses_all_probabilities_and_is_differentiable(self):
        # Keep argmax unchanged while moving probability mass: a continuous
        # embedding must change, whereas the old selected embedding would not.
        with torch.no_grad():
            self.model.consensus[-1].weight.zero_()
            self.model.consensus[-1].bias.zero_()
            self.model.consensus[-1].bias[0] = 0.02
        h = self.hidden.clone()
        p1, e1 = self.model.consensus_features(h, h)
        with torch.no_grad():
            self.model.consensus[-1].bias[0] = 0.01
        p2, e2 = self.model.consensus_features(h, h)
        self.assertTrue(torch.equal(p1.argmax(-1), p2.argmax(-1)))
        self.assertFalse(torch.equal(e1, e2))
        e2.square().mean().backward()
        self.assertTrue((self.model.consensus_embedding.weight.grad.abs().sum(-1) > 0).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.consensus.parameters()))

    def test_individual_target_stops_at_hidden_both_projectors_learn(self):
        hidden = torch.rand(2, 3, 16, requires_grad=True)
        latent = torch.rand(2, 3, 3, 16, requires_grad=True)
        self.model.individual_contrastive_loss(latent, hidden).backward()
        self.assertIsNone(hidden.grad)
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        for projector in (self.model.projector, self.model.target_projector):
            self.assertTrue(all(p.requires_grad for p in projector.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))
        self.model.zero_grad(set_to_none=True)
        self.args.ritrl_rec_weight = self.args.ritrl_cons_weight = 0.
        self.args.ritrl_cl_weight = 1.
        loss, *_ = self.model.auxiliary_loss(self.obs, self.previous, self.hidden, self.hidden, 2)
        loss.backward()
        for module in (self.model.encoder, self.model.decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.target_encoder.parameters()))

    def test_individual_target_uses_online_hidden_and_is_not_ema_updated(self):
        target_projector_before = {name: p.detach().clone() for name, p in self.model.target_projector.named_parameters()}
        with torch.no_grad():
            for p in self.model.target_encoder.parameters():
                p.add_(10.)
            for p in self.model.projector.parameters():
                p.add_(0.1)
        self.model.update_targets()
        for name, p in self.model.target_projector.named_parameters():
            torch.testing.assert_close(p, target_projector_before[name], rtol=0, atol=0)
        # Perturbing only the EMA encoder cannot alter L_ind.
        _, _, _, _, first = self.model.auxiliary_loss(self.obs, self.previous, self.hidden, self.hidden, 2)
        with torch.no_grad():
            for p in self.model.target_encoder.parameters():
                p.zero_()
        _, _, _, _, second = self.model.auxiliary_loss(self.obs, self.previous, self.hidden, self.hidden, 2)
        torch.testing.assert_close(first['ritrl_ind_loss'], second['ritrl_ind_loss'], rtol=0, atol=0)

    def test_consensus_gradient_boundary_and_targets(self):
        self.args.ritrl_rec_weight = self.args.ritrl_cl_weight = 0.
        loss, *_ = self.model.auxiliary_loss(self.obs, self.previous, self.hidden, self.hidden, 2)
        loss.backward()
        for module in (self.model.encoder, self.model.decoder, self.model.query,
                       self.model.target_encoder, self.model.target_consensus):
            self.assertTrue(all(p.grad is None or not p.grad.any() for p in module.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.any() for p in self.model.consensus.parameters()))

    def test_ema_and_center(self):
        old = next(self.model.target_encoder.parameters()).clone()
        with torch.no_grad():
            next(self.model.encoder.parameters()).add_(1.)
        self.model.update_targets(torch.ones_like(self.model.center))
        torch.testing.assert_close(next(self.model.target_encoder.parameters()),
                                   old + (1 - self.args.ritrl_ema))
        torch.testing.assert_close(self.model.center, torch.full_like(self.model.center, 0.1))

    def rollout(self, policy):
        buffer = TrajectoryBuffer(device='cpu')
        s = torch.rand(2, 3, 13 + (2 if policy.input_args.obs_add_action else 0))
        for t in range(4):
            policy.reset_gru_hidden('train', [t == 0 or t == 2, t == 0])
            dist = policy.act(s, 'train')
            a = torch.stack([dist['branch1'].sample(), dist['branch2'].sample()], -1)
            logp = torch.stack([dist['branch1'].log_prob(a[..., 0]),
                                dist['branch2'].log_prob(a[..., 1])], -1)
            hidden = policy.last_gru_hidden_input
            torch.testing.assert_close(policy.get_logp_with_hidden(s, a, hidden), logp)
            if t == 2:
                self.assertFalse(hidden[0, :, :-2].any())
                self.assertTrue((hidden[0, :, -2:] == -1).all())
            policy.record_actions(a, 'train')
            s1 = torch.rand_like(s)
            done = torch.zeros(2, 3, dtype=torch.bool)
            if t == 1:
                done[0] = True
            buffer.store(s, a, torch.rand(2, 3), s1, done, logp, gru_h=hidden,
                         user_visibility=torch.ones(2, 3, 2, dtype=torch.bool),
                         uav_visibility=torch.eye(3, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1))
            s = s1
        return buffer.retrieve()

    def test_rollout_update_checkpoint_and_phase_isolation(self):
        for mappo, obs_action in [(False, False), (True, True)]:
            args = arguments('--ritrl_hidden_dim', '16', '--ritrl_sequence_length', '2')
            args.MAPPO, args.obs_add_action = mappo, obs_action
            policy = agent(args)
            trajs = self.rollout(policy)
            old = policy.ritrl.observation_head.weight.detach().clone()
            old_target_projection = next(policy.ritrl.target_projector.parameters()).detach().clone()
            result = policy.updateAgent(trajs)
            self.assertTrue(all(torch.isfinite(torch.tensor(result))))
            self.assertFalse(torch.equal(old, policy.ritrl.observation_head.weight))
            self.assertFalse(torch.equal(old_target_projection, next(policy.ritrl.target_projector.parameters())))
            self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))
            train_state = policy._gru_hidden['train'].clone()
            policy.act(trajs[0]['s'][:1], 'test')
            torch.testing.assert_close(train_state, policy._gru_hidden['train'])
            with tempfile.TemporaryDirectory() as directory:
                policy.save_nets(directory, iter=1)
                restored = agent(args)
                restored.load_nets(directory, iter=1)
                for key, value in policy.actors.state_dict().items():
                    torch.testing.assert_close(value, restored.actors.state_dict()[key])

    def test_pretraining_stages(self):
        args = arguments('--ritrl_rec_warmup', '1', '--ritrl_cl_warmup', '1')
        policy = agent(args)
        for stage in range(3):
            trajs = self.rollout(policy)
            policy.updateAgent(trajs)
            self.assertEqual(int(policy.ritrl.aux_updates), stage + 1)

    def test_argument_validation_and_default_on(self):
        with patch.object(sys, 'argv', ['test']):
            self.assertTrue(parse_args()[0].use_ritrl)
        self.assertFalse(arguments('--use_ritrl', 'False').use_ritrl)
        self.assertTrue(arguments('--use_ritrl', 'True').use_ritrl)
        self.assertTrue(self.args.ues_GRU)
        for extra in [('--uav_num', '1'), ('--algo', 'DPPO'),
                      ('--ritrl_cl_temperature', '0'), ('--ritrl_ema', '1'),
                      ('--ritrl_sequence_length', '0'), ('--ritrl_lambda0', '0'), ('--ritrl_lambda0', '1')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(*extra)


if __name__ == '__main__':
    unittest.main()
