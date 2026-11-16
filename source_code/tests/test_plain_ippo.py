import ast
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.models import MLP
from algorithms.plain_ippo import CategoricalActor0

class TestPlainIPPO(unittest.TestCase):
    def test_reference_initialization_outputs_and_gradients(self):
        reference = Path(__file__).resolve().parents[3] / 'DRL_dyna_AoI_uav_JSAC3/source_code/algorithms/models.py'
        tree = ast.parse(reference.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CategoricalActor0')
        scope = {'torch': torch, 'nn': nn, 'MLP': MLP}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(reference), 'exec'), scope)
        kwargs = dict(network=MLP, sizes=[13,64,64], branchs=[9,4], have_last_branch=False, comm=False, snrmap_features=0)
        torch.manual_seed(19)
        expected = [scope['CategoricalActor0'](**kwargs) for _ in range(3)]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(19)
        actual = [CategoricalActor0(**kwargs) for _ in range(3)]
        torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0, atol=0)
        obs = torch.rand(4,13)
        for ref, actor in zip(expected, actual):
            self.assertEqual(set(ref.state_dict()), set(actor.state_dict()))
            for key, value in ref.state_dict().items():
                torch.testing.assert_close(value, actor.state_dict()[key], rtol=0, atol=0)
            ref_probs = ref(obs, 0, [])
            probs = actor(obs)
            torch.testing.assert_close(probs, ref_probs, rtol=0, atol=0)
            loss = -probs[:,0].log().mean()
            loss.backward()
            self.assertIsNotNone(actor.network[0].weight.grad)
            self.assertIsNone(actor.lstm.weight_ih_l0.grad)

if __name__ == '__main__':
    unittest.main()
