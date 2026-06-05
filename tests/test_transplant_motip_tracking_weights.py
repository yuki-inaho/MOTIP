import torch

from tools.transplant_motip_tracking_weights import (
    EMBED_TO_WORD_PREFIX,
    WORD_TO_EMBED_KEY,
    apply_weight_transfer,
)


def test_apply_weight_transfer_copies_exact_matching_tracking_key():
    source = {"trajectory_modeling.norm.weight": torch.tensor([1.0, 2.0])}
    target = {"trajectory_modeling.norm.weight": torch.zeros(2)}

    report = apply_weight_transfer(
        name="unit",
        source_state=source,
        target_state=target,
        source_label="source",
        include_prefixes=["trajectory_modeling."],
    )

    assert len(report.copied_exact) == 1
    assert target["trajectory_modeling.norm.weight"].tolist() == [1.0, 2.0]


def test_apply_weight_transfer_partially_copies_word_to_embed_vocab_and_unknown():
    source = {WORD_TO_EMBED_KEY: torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)}
    target = {WORD_TO_EMBED_KEY: torch.full((3, 6), -1.0)}

    report = apply_weight_transfer(
        name="unit",
        source_state=source,
        target_state=target,
        source_label="source",
        include_prefixes=["id_decoder."],
        vocab_policy="copy-overlap",
        copy_unknown=True,
    )

    copied = target[WORD_TO_EMBED_KEY]
    assert len(report.copied_partial) == 1
    assert report.copied_partial[0].copied_primary == 3
    assert torch.equal(copied[:, :3], source[WORD_TO_EMBED_KEY][:, :3])
    assert torch.equal(copied[:, -1], source[WORD_TO_EMBED_KEY][:, -1])
    assert torch.equal(copied[:, 3:5], torch.full((3, 2), -1.0))


def test_apply_weight_transfer_partially_copies_embed_to_word_vocab_and_unknown():
    key = f"{EMBED_TO_WORD_PREFIX}0.weight"
    source = {key: torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)}
    target = {key: torch.full((6, 3), -1.0)}

    report = apply_weight_transfer(
        name="unit",
        source_state=source,
        target_state=target,
        source_label="source",
        include_prefixes=["id_decoder."],
        vocab_policy="copy-overlap",
        copy_unknown=True,
    )

    copied = target[key]
    assert len(report.copied_partial) == 1
    assert report.copied_partial[0].copied_primary == 3
    assert torch.equal(copied[:3, :], source[key][:3, :])
    assert torch.equal(copied[-1, :], source[key][-1, :])
    assert torch.equal(copied[3:5, :], torch.full((2, 3), -1.0))


def test_apply_weight_transfer_skips_non_vocab_shape_mismatch():
    source = {"id_decoder.self_attn_layers.0.out_proj.weight": torch.ones(2, 3)}
    target = {"id_decoder.self_attn_layers.0.out_proj.weight": torch.zeros(4, 3)}

    report = apply_weight_transfer(
        name="unit",
        source_state=source,
        target_state=target,
        source_label="source",
        include_prefixes=["id_decoder."],
    )

    assert len(report.skipped) == 1
    assert report.skipped[0].reason == "shape mismatch"
    assert torch.equal(target["id_decoder.self_attn_layers.0.out_proj.weight"], torch.zeros(4, 3))
