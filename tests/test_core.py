"""Edge-case checks for smalign beyond the public smoke tests.

Covers truncation boundaries, tokenizers without PAD or EOS, PAD == EOS,
mixed sequence lengths, row order, agreement with a reference computation and
with Hugging Face's own loss, and that a few DPO steps move the policy the
right way while leaving the reference untouched.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from smalign import batching, losses
from smalign.batching import IGNORE_INDEX
from tests.toy_lm import ToyCausalLM, ToyTokenizer


@pytest.fixture(scope="module")
def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=ToyTokenizer().vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


# ---------------------------------------------------------------- batching


def test_eos_is_appended_then_truncated():
    tok = ToyTokenizer()
    prompt, response = "ab: ", "xyz"  # 4 prompt + 3 response + 1 eos = 8
    full = adapter.build_sft_batch(tok, [prompt], [response], max_length=8)
    assert full["labels"][0, -1] == tok.eos_token_id

    cut = adapter.build_sft_batch(tok, [prompt], [response], max_length=7)
    assert cut["input_ids"].shape[1] == 7
    assert tok.eos_token_id not in cut["labels"][0].tolist()
    assert cut["labels"][0, 4:].tolist() == tok.encode("xyz")


def test_prompt_longer_than_max_length_has_no_response_tokens(tiny_qwen):
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["a very long prompt"], ["ok"], max_length=5)
    assert batch["labels"].eq(IGNORE_INDEX).all()
    logps = adapter.compute_response_logprobs(tiny_qwen, tok, ["a very long prompt"], ["ok"], max_length=5)
    assert logps.tolist() == [0.0]


def test_padding_is_masked_by_position_when_pad_equals_eos():
    tok = ToyTokenizer()
    tok.pad_token_id = tok.eos_token_id  # Qwen2.5 style
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "abcd"], max_length=16)
    # Short row: prompt 3, "a" 1, eos 1, then padding with the same id as eos.
    assert batch["labels"][0].tolist()[3:5] == [tok.encode("a")[0], tok.eos_token_id]
    assert batch["labels"][0, 5:].eq(IGNORE_INDEX).all()
    assert batch["attention_mask"][0].tolist() == [1] * 5 + [0] * 3


def test_tokenizer_without_pad_falls_back_to_eos():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "abcd"], max_length=16)
    assert batch["input_ids"][0, 5:].eq(tok.eos_token_id).all()


def test_tokenizer_without_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ab"], max_length=16)
    assert batch["labels"][0].tolist() == [IGNORE_INDEX] * 3 + tok.encode("ab")


def test_max_prompt_length_keeps_the_end_of_the_prompt():
    tok = ToyTokenizer()
    ids, labels = batching.tokenize_example(tok, "abcdef", "z", max_length=16, max_prompt_length=2)
    assert ids == tok.encode("ef") + tok.encode("z") + [tok.eos_token_id]
    assert labels[:2] == [IGNORE_INDEX] * 2


def test_dpo_batch_sides_are_independent_and_ordered():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["A: ", "Bb: "], ["x", "yy"], ["long one", "z"], max_length=32)
    expected_chosen = adapter.build_sft_batch(tok, ["A: ", "Bb: "], ["x", "yy"], max_length=32)
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(batch["chosen"][key], expected_chosen[key])
    assert batch["rejected"]["input_ids"].shape[1] == len("A: long one") + 1


# ---------------------------------------------------------------- log-probs


def manual_logps(model, tok, prompt, response):
    ids = tok.encode(prompt) + tok.encode(response) + [tok.eos_token_id]
    n_prompt = len(tok.encode(prompt))
    logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    logp = F.log_softmax(logits[:-1], dim=-1)
    targets = torch.tensor(ids[1:])
    per_token = logp.gather(-1, targets[:, None]).squeeze(-1)
    return per_token[n_prompt - 1 :].sum()


def test_logprobs_match_manual_computation_and_ignore_padding(tiny_qwen):
    tok = ToyTokenizer()
    prompts = ["Q: hi ", "Question: what? ", "x"]
    responses = ["a much longer answer here", "no", "y"]
    batched = adapter.compute_response_logprobs(tiny_qwen, tok, prompts, responses, max_length=64)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        alone = adapter.compute_response_logprobs(tiny_qwen, tok, [p], [r], max_length=64)
        torch.testing.assert_close(batched[i], alone[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched[i], manual_logps(tiny_qwen, tok, p, r), rtol=1e-5, atol=1e-5)


def test_row_order_is_preserved(tiny_qwen):
    tok = ToyTokenizer()
    prompts, responses = ["A: ", "Bbb: ", "C: "], ["one", "two words", "3"]
    forward = adapter.compute_response_logprobs(tiny_qwen, tok, prompts, responses, max_length=32)
    backward = adapter.compute_response_logprobs(tiny_qwen, tok, prompts[::-1], responses[::-1], max_length=32)
    torch.testing.assert_close(forward, backward.flip(0), rtol=1e-5, atol=1e-5)


def test_sft_loss_matches_hugging_face_token_mean(tiny_qwen):
    tok = ToyTokenizer()
    prompts, responses = ["Q: ", "Longer prompt: "], ["answer", "b"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=32)
    ours = adapter.compute_sft_loss(tiny_qwen, tok, prompts, responses, max_length=32)
    hf = tiny_qwen(**batch).loss
    torch.testing.assert_close(ours, hf, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- DPO


def test_dpo_loss_from_logps_matches_formula():
    pc, pr = torch.tensor([-1.0, -4.0]), torch.tensor([-3.0, -2.0])
    rc, rr = torch.tensor([-2.0, -4.0]), torch.tensor([-2.0, -2.5])
    beta = 0.5
    expected = [
        -math.log(1 / (1 + math.exp(-beta * ((a - c) - (b - d)))))
        for a, b, c, d in zip(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())
    ]
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(loss, torch.tensor(sum(expected) / 2))


def test_dpo_reward_accuracy_is_strict_when_tied():
    v = torch.tensor([-1.0, -2.0])
    _, stats = losses.dpo_loss_from_logps(v, v, v, v, beta=0.1)
    assert stats["reward_correct"].sum() == 0
    assert stats["reward_margin"].abs().sum() == 0


def test_dpo_steps_increase_margin_and_leave_reference_alone():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size)
    ref_before = reference.logit_bias.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    args = (tok, ["Pick: "], ["a"], ["b"])

    first = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=12)
    for _ in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=12)
        loss.backward()
        optimizer.step()
    last = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=12)

    assert last < first
    assert torch.equal(reference.logit_bias, ref_before)
    assert reference.logit_bias.grad is None
    bias = policy.logit_bias.detach()
    assert bias[tok.vocab["a"]] > bias[tok.vocab["b"]]


def test_concat_pair_batch_matches_separate_forward(tiny_qwen):
    tok = ToyTokenizer()
    dpo = batching.build_dpo_batch(tok, ["P: ", "Q: "], ["short", "s"], ["much longer text", "t"], max_length=32)
    joined = batching.concat_pair_batch(dpo, tok.pad_token_id)
    together = losses.sequence_logprobs(tiny_qwen, joined)
    separate = torch.cat([losses.sequence_logprobs(tiny_qwen, dpo[s]) for s in ("chosen", "rejected")])
    torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-5)
