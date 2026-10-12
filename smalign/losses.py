"""Response-only log probabilities, the SFT loss, and the DPO objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from smalign.batching import IGNORE_INDEX


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def token_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log p(label_t | tokens < t) and the response mask, both shaped (B, T-1).

    Masked positions have log-probability 0. Log-softmax runs in fp32 even
    under fp16 autocast, since summed fp16 log-probs lose too much precision.
    """

    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)

    # cross_entropy with ignore_index avoids materialising a full log-softmax copy.
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(targets)
    return -nll * mask, mask


def sequence_logprobs(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """One summed response log-probability per row, shape (B,)."""

    logps, _ = token_logprobs(model, batch)
    return logps.sum(dim=-1)


def sft_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Mean negative log-likelihood over all response tokens in the batch.

    Returns the loss and the token count (for exact token-weighted validation
    averages). This is the per-token reduction allowed by the assignment.
    """

    logps, mask = token_logprobs(model, batch)
    n_tokens = int(mask.sum())
    loss = -logps.sum() / max(n_tokens, 1)
    return loss, n_tokens


def dpo_loss_from_logps(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batch-mean DPO loss plus detached per-example implicit rewards."""

    reference_chosen = reference_chosen.to(policy_chosen.device).detach()
    reference_rejected = reference_rejected.to(policy_chosen.device).detach()

    chosen_logratio = policy_chosen - reference_chosen
    rejected_logratio = policy_rejected - reference_rejected
    logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = (beta * chosen_logratio).detach()
    rejected_rewards = (beta * rejected_logratio).detach()
    stats = {
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_correct": (chosen_rewards > rejected_rewards).float(),
        "reward_margin": chosen_rewards - rejected_rewards,
    }
    return loss, stats


@torch.no_grad()
def reference_logprobs(reference_model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return sequence_logprobs(reference_model, batch)
