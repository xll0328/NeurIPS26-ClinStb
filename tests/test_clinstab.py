import unittest
from types import SimpleNamespace

import torch

from models.ClinStab import Model
from utils.clinstab_objective import projected_gradient_attack, stability_objective


def config(channels=5, classes=3):
    return SimpleNamespace(
        task_name="classification",
        seq_len=32,
        enc_in=channels,
        d_model=24,
        n_heads=8,
        dropout=0.0,
        num_class=classes,
        patch_len_list="2,4,8",
        scdm_alpha=2.0,
        e_layers=2,
    )


class ClinStabTest(unittest.TestCase):
    def test_forward_gradient_and_routing(self):
        model = Model(config())
        x = torch.randn(4, 31, 5, requires_grad=True)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(len(model.routing_indices()), 6)
        self.assertTrue(all(tuple(i.shape) == (4, 2) for i in model.routing_indices()))
        logits.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_one_channel_edge_case(self):
        model = Model(config(channels=1, classes=2))
        logits = model(torch.randn(2, 32, 1))
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_objective_and_attack(self):
        model = Model(config(channels=3, classes=2))
        x = torch.randn(3, 32, 3)
        labels = torch.tensor([0, 1, 1])
        model.train()
        perturbed = projected_gradient_attack(
            model, x, labels, epsilon=0.1, step_size=0.03, steps=2
        )
        self.assertTrue(model.training)
        self.assertLessEqual(float((perturbed - x).abs().max()), 0.100001)
        clean_logits = model(x)
        perturbed_logits = model(perturbed)
        loss, terms = stability_objective(
            clean_logits,
            perturbed_logits,
            labels,
            teacher_logits=clean_logits.detach(),
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(
            set(terms), {"clean_ce", "robust_ce", "consistency", "knowledge", "total"}
        )


if __name__ == "__main__":
    unittest.main()
