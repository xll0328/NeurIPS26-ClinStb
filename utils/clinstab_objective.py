"""Stability losses and bounded perturbations used by ClinStab training.

The helpers are intentionally independent of the experiment runner so they can
be used with a custom dataloader or with the existing ``run_rob.py`` entry
point.  They implement the four terms described in the paper: clean CE,
perturbed CE, probability consistency, and teacher KL distillation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def projected_gradient_attack(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.20,
    step_size: float = 0.01,
    steps: int = 10,
    random_start: bool = True,
) -> torch.Tensor:
    """Generate an L-infinity bounded PGD perturbation.

    The attack maximizes cross-entropy while keeping the perturbation in the
    normalized input-space box ``[-epsilon, epsilon]``.  Model parameters are
    not modified and the returned tensor is detached for the outer update.
    """

    if epsilon < 0 or step_size <= 0 or steps < 1:
        raise ValueError("epsilon must be non-negative; step_size and steps must be positive")
    was_training = model.training
    model.eval()
    try:
        original = inputs.detach()
        if random_start and epsilon > 0:
            adversarial = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        else:
            adversarial = original.clone()

        for _ in range(int(steps)):
            adversarial.requires_grad_(True)
            logits = model(adversarial)
            loss = F.cross_entropy(logits, labels.long())
            gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
            with torch.no_grad():
                adversarial = adversarial + float(step_size) * gradient.sign()
                delta = torch.clamp(adversarial - original, min=-epsilon, max=epsilon)
                adversarial = (original + delta).detach()

        return adversarial
    finally:
        model.train(was_training)


def stability_objective(
    clean_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    lambda_rob: float = 1.0,
    lambda_con: float = 0.5,
    lambda_know: float = 0.5,
    temperature: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute ClinStab's composite objective and its individual terms."""

    if clean_logits.shape != perturbed_logits.shape:
        raise ValueError("clean_logits and perturbed_logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    clean_ce = F.cross_entropy(clean_logits, labels.long())
    robust_ce = F.cross_entropy(perturbed_logits, labels.long())
    clean_prob = clean_logits.softmax(dim=-1)
    perturbed_prob = perturbed_logits.softmax(dim=-1)
    consistency = F.mse_loss(perturbed_prob, clean_prob)

    if teacher_logits is None:
        knowledge = clean_logits.new_zeros(())
    else:
        if teacher_logits.shape != perturbed_logits.shape:
            raise ValueError("teacher_logits must have the same shape as perturbed_logits")
        teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        knowledge = F.kl_div(
            F.log_softmax(perturbed_logits / temperature, dim=-1),
            teacher_prob,
            reduction="batchmean",
        ) * (temperature**2)

    total = clean_ce + lambda_rob * robust_ce + lambda_con * consistency + lambda_know * knowledge
    terms = {
        "clean_ce": clean_ce.detach(),
        "robust_ce": robust_ce.detach(),
        "consistency": consistency.detach(),
        "knowledge": knowledge.detach(),
        "total": total.detach(),
    }
    return total, terms
