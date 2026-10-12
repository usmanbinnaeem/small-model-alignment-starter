"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.

The implementation lives in ``smalign/``: batching in ``smalign/batching.py``
and log-probabilities and losses in ``smalign/losses.py``. ``compute_sft_loss``
uses the per-token reduction: the mean negative log-likelihood over all
unmasked response tokens in the batch.
"""

from __future__ import annotations

from smalign import batching, losses


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return batching.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = batching.build_lm_batch(tokenizer, prompts, responses, max_length)
    return losses.sequence_logprobs(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    batch = batching.build_lm_batch(tokenizer, prompts, responses, max_length)
    loss, _ = losses.sft_loss(model, batch)
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return batching.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    loss, _ = losses.dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    return loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    batch = batching.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_chosen = losses.sequence_logprobs(policy_model, batch["chosen"])
    policy_rejected = losses.sequence_logprobs(policy_model, batch["rejected"])
    reference_chosen = losses.reference_logprobs(reference_model, batch["chosen"])
    reference_rejected = losses.reference_logprobs(reference_model, batch["rejected"])
    loss, _ = losses.dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    return loss
