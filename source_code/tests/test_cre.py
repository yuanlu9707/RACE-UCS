import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_ritrl import arguments, agent
from algorithms.cre import CRE, ReturnNormalizer
from algorithms.algo.buffer import TrajectoryBuffer


class TestCRE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.args = arguments('--use_cre', 'True', '--cre_batch_size', '12')
        self.model = CRE(64, 80, [9, 4], self.args)

    def test_gaussian_moments(self):
        mu = torch.tensor([0.3, -0.6])
        logvar = torch.tensor([0.4, -0.3])
        mean, variance = CRE.moments(mu, logvar)
        q = torch.distributions.Normal(mu, (0.5 * logvar).exp())
        sample = q.sample((120000,))
        psi = (q.log_prob(sample) - torch.distributions.Normal(0., 1.).log_prob(sample)).sum(-1)
        torch.testing.assert_close(mean, psi.mean(), rtol=0.03, atol=0.01)
        torch.testing.assert_close(variance, psi.var(), rtol=0.04, atol=0.01)

    def batch(self):
        return (torch.randn(12, 64), torch.randn(12, 80), torch.randn(12, 80),
                torch.randn(12, 64), torch.randn(12, 1), torch.ones(12, 1),
                torch.arange(12) // 3, torch.stack([torch.randint(9, (12,)), torch.randint(4, (12,))], -1),
                torch.zeros(12, 1, dtype=torch.bool))

    def test_three_term_loss_and_separate_score_gradient(self):
        self.args.cre_cmin, self.args.cre_cmax = 100., 101.
        self.args.cre_alpha_I = 2.5  # non-unit weight catches placement errors
        batch = self.batch()
        for value in batch[:4]:
            value.requires_grad_(True)
        loss, metrics = self.model.loss(*batch)
        torch.testing.assert_close(loss.detach(), metrics['cre_pred_loss'] + self.args.cre_alpha_I *
                                   (metrics['cre_upper_loss'] + metrics['cre_lower_loss']))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.scorer_parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.target_state.parameters()))
        self.assertTrue(all(value.grad is None for value in batch[:4]))
        self.assertFalse(any('ctx_loss' in key or 'nce_loss' in key for key in metrics))
        self.model.zero_grad(set_to_none=True)
        # Scorer supervision does not depend on a Cantelli constraint being active.
        self.args.cre_cmin, self.args.cre_cmax = -1000., 1000.
        _, inactive_metrics = self.model.loss(*batch)
        self.assertEqual(float(inactive_metrics['cre_upper_loss'] + inactive_metrics['cre_lower_loss']), 0.)
        score_loss, active_count = self.model.scorer_loss(batch[1], batch[2], batch[4], batch[6], batch[5])
        self.assertGreater(int(active_count), 0)
        score_loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.model.history_score.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.main_parameters()))

    def test_zero_calibration_weight_and_sample_averaging(self):
        self.args.cre_alpha_I = 0.
        batch = self.batch()
        # Hold stochastic latent samples fixed while duplicating every UAV/time
        # sample: an averaged loss must stay constant instead of doubling.
        with patch('torch.randn_like', side_effect=lambda value: torch.zeros_like(value)):
            first, metrics = self.model.loss(*batch)
            duplicated = tuple(torch.cat([value, value], 0) for value in batch)
            second, _ = self.model.loss(*duplicated)
        torch.testing.assert_close(first.detach(), metrics['cre_pred_loss'])
        torch.testing.assert_close(first, second)

    def test_association_is_positive_softmax_probability(self):
        from torch.nn import functional as F
        b = self.batch()
        gamma, context, active = self.model.association(b[1], b[2], b[4], b[6], b[5])
        history = F.normalize(self.model.history_score(torch.cat([self.model.context(b[2]), b[4]], -1)), dim=-1)
        keys = F.normalize(self.model.current_score(context), dim=-1)
        for i in range(12):
            negative_ids = torch.where(b[6] != b[6][i])[0]
            indices = torch.cat([torch.tensor([i]), negative_ids])
            scores = (history[i] * keys[indices]).sum(-1) / self.args.cre_temperature
            expected = scores.softmax(0)[0]
            torch.testing.assert_close(gamma[i], expected)
        self.assertTrue(active.all())
        self.assertTrue(((gamma > 0) & (gamma < 1)).all())

    def test_no_same_episode_negatives_and_terminal_prediction(self):
        batch = list(self.batch())
        batch[6] = torch.zeros(12, dtype=torch.long)
        gamma, _, active = self.model.association(batch[1], batch[2], batch[4], batch[6], batch[5])
        self.assertFalse(active.any())
        self.assertFalse(gamma.any())
        self.model.enqueue(batch[1], torch.ones(12, dtype=torch.long))
        _, _, active = self.model.association(batch[1], batch[2], batch[4], batch[6], batch[5])
        self.assertTrue(active.all())
        batch[-1].fill_(True)
        loss, metrics = self.model.loss(*batch)
        self.assertEqual(float(metrics['cre_pred_loss']), 0.)
        self.assertTrue(torch.isfinite(loss))

    def test_bounded_calibration(self):
        b = self.batch()
        mean, variance = self.model.moments(*self.model.distribution(b[0], b[7]))
        gamma, ctx, _ = self.model.association(b[1], b[2], b[4], b[6], b[5])
        std = (variance + self.args.cre_epsilon).sqrt()
        actual = self.model.calibrated_mean(mean, variance, ctx, gamma)
        expected = mean + self.args.cre_rho * gamma * std * torch.tanh(self.model.calibration(
            torch.cat([ctx, mean[:, None]], -1)).squeeze(-1))
        torch.testing.assert_close(actual, expected)
        self.assertTrue(((actual - mean).abs() <= self.args.cre_rho * gamma * std + 1e-6).all())
        torch.testing.assert_close(self.model.calibrated_mean(mean, variance, ctx, torch.zeros_like(gamma)), mean,
                                   rtol=0, atol=0)
        # Fixed H input: halving gamma must halve the offset, not change its direction.
        half = self.model.calibrated_mean(mean, variance, ctx, gamma / 2)
        torch.testing.assert_close(half - mean, (actual - mean) / 2, atol=1e-6, rtol=1e-4)

    def test_shared_reward_scale_running_moments(self):
        normalizer = ReturnNormalizer(0.01)
        normalizer.update(torch.tensor([1., 3.]))
        torch.testing.assert_close(normalizer.count, torch.tensor(3., dtype=torch.float64))
        torch.testing.assert_close(normalizer.mean, torch.tensor(4/3, dtype=torch.float64))
        torch.testing.assert_close(normalizer.scale(), torch.tensor((17/9)**0.5, dtype=torch.float64))
        rewards = torch.tensor([0., 2., 4.])
        normalized = normalizer(rewards)
        torch.testing.assert_close(normalized, rewards / ((17/9)**0.5))
        self.assertTrue((normalized >= 0).all())
        # Sequential parallel-moment merges match a single update.
        merged = ReturnNormalizer(0.01)
        merged.update(torch.tensor([1.]))
        merged.update(torch.tensor([3.]))
        torch.testing.assert_close(merged.scale(), normalizer.scale())

    def test_agent_rewards_update_reset_and_checkpoint(self):
        policy = agent(self.args)
        buffer = TrajectoryBuffer('cpu')
        s = torch.rand(2, 3, 13)
        for step in range(4):
            policy.reset_gru_hidden('train', [step in (0, 2), step == 0])
            captured = []
            hook = policy.ritrl.fusion.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
            try:
                dist = policy.act(s, 'train')
            finally:
                hook.remove()
            # CRE must consume the exact same e_c supplied to Actor fusion.
            width = policy.input_args.ritrl_consensus_embed_dim
            torch.testing.assert_close(policy._cre_current[1][..., -width:], captured[0][..., -width:])
            action = torch.stack([dist['branch1'].sample(), dist['branch2'].sample()], -1)
            logp = torch.stack([dist['branch1'].log_prob(action[..., 0]), dist['branch2'].log_prob(action[..., 1])], -1)
            policy.record_actions(action, 'train')
            next_obs = torch.rand_like(s)
            done = torch.zeros(2, 3, dtype=torch.bool)
            if step == 1:
                done[0] = True
            external = torch.rand(2, 3)
            mixed = torch.as_tensor(policy.cre_transition(next_obs, action, external, done))
            bonus = mixed - external
            torch.testing.assert_close(bonus[:, :1].expand_as(bonus), bonus)
            self.assertTrue((bonus >= 0).all())
            if step == 1:
                self.assertFalse(policy._cre_returns[0].any())
            if step == 2:
                self.assertFalse(policy.last_cre_data[0, :, -2].any())
                self.assertTrue(policy.last_cre_data[1, :, -2].all())
            buffer.store(s, action, mixed, next_obs, done, logp,
                         gru_h=policy.last_gru_hidden_input, cre_data=policy.last_cre_data)
            s = next_obs
        trajs = buffer.retrieve()
        torch.testing.assert_close(trajs[0].getFraction(2, 1).cre_data, trajs[0].cre_data[1:3])
        before = policy.cre.posterior[0].weight.clone()
        score_before = policy.cre.history_score[0].weight.clone()
        embedding_before = policy.ritrl.consensus_embedding.weight.detach().clone()
        direct_obs_before = policy.ritrl.policy_obs.networks[0][0].weight.detach().clone()
        policy_mlp_before = policy.ritrl.policy_mlp.networks[0][0].weight.detach().clone()
        fixed_rewards = trajs[0]['r'].clone()
        # Simulate gradients left behind by the preceding rollout's auxiliary
        # update: freezing must clear them before the next PPO optimizer step.
        for name, parameter in policy.ritrl.named_parameters():
            if parameter.requires_grad and not name.startswith(('consensus_embedding.', 'fusion.', 'policy_obs.', 'policy_mlp.', 'heads.')):
                parameter.grad = torch.ones_like(parameter)
        original_step = policy.optimizer_pi.step
        checked_updates = []
        def check_policy_step(*args, **kwargs):
            allowed = ('consensus_embedding.', 'fusion.', 'policy_obs.', 'policy_mlp.', 'heads.')
            frozen = {name: p.detach().clone() for name, p in policy.ritrl.named_parameters()
                      if not name.startswith(allowed)}
            for name, p in policy.ritrl.named_parameters():
                if name in frozen:
                    self.assertFalse(p.requires_grad)
                    self.assertIsNone(p.grad)
            self.assertTrue(policy.ritrl.consensus_embedding.weight.requires_grad)
            for module in (policy.ritrl.policy_obs, policy.ritrl.policy_mlp):
                self.assertTrue(all(p.requires_grad for p in module.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
            result = original_step(*args, **kwargs)
            for name, p in policy.ritrl.named_parameters():
                if name in frozen:
                    torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
            checked_updates.append(True)
            return result
        with patch.object(policy.optimizer_pi, 'step', side_effect=check_policy_step):
            policy.updateAgent(trajs)
        self.assertTrue(checked_updates)
        self.assertFalse(torch.equal(embedding_before, policy.ritrl.consensus_embedding.weight))
        self.assertFalse(torch.equal(direct_obs_before, policy.ritrl.policy_obs.networks[0][0].weight))
        self.assertFalse(torch.equal(policy_mlp_before, policy.ritrl.policy_mlp.networks[0][0].weight))
        self.assertTrue(all(p.requires_grad for p in policy.ritrl.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in policy.ritrl.target_consensus.parameters()))
        self.assertFalse(torch.equal(before, policy.cre.posterior[0].weight))
        self.assertFalse(torch.equal(score_before, policy.cre.history_score[0].weight))
        torch.testing.assert_close(fixed_rewards, trajs[0]['r'], rtol=0, atol=0)
        main_ids = {id(p) for p in policy.cre.main_parameters()}
        self.assertTrue(all(id(p) not in main_ids for p in policy.cre.scorer_parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            policy.save_nets(directory)
            restored = agent(self.args)
            restored.load_nets(directory)
            for key, value in policy.cre.state_dict().items():
                torch.testing.assert_close(value, restored.cre.state_dict()[key])
            legacy = {key: value for key, value in policy.cre.state_dict().items()
                      if not key.startswith('reward_normalizer.')}
            torch.save(legacy, str(Path(directory) / 'Models' / '0_cre.pt'))
            migrated = agent(self.args)
            migrated.load_nets(directory)
            self.assertEqual(float(migrated.cre.reward_normalizer.count), 1.)
            torch.testing.assert_close(migrated.cre.posterior[0].weight, policy.cre.posterior[0].weight)
        # Policy inference in evaluation must not call the exploration module.
        with patch.object(policy.cre, 'intrinsic_reward', side_effect=AssertionError('CRE used at test')):
            policy.act(s, 'test')

    def test_argument_validation(self):
        self.assertFalse(arguments().use_cre)
        self.assertTrue(arguments('--use_cre', 'True').use_cre)
        self.assertFalse(arguments('--use_cre', 'False').use_cre)
        for extra in [('--use_ritrl', 'False'), ('--cre_tail_probability', '1'),
                      ('--cre_cmin', '20'), ('--cre_batch_size', '0'), ('--cre_lr', 'nan'),
                      ('--cre_score_lr', '0'), ('--cre_norm_discount', '1'), ('--cre_norm_min_std', '0')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments('--use_cre', 'True', *extra)


if __name__ == '__main__':
    unittest.main()
