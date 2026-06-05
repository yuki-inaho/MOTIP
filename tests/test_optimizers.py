import torch

from models.optim import AutoMuonWithAuxAdam, build_optimizer, set_optimizer_mode


def test_build_optimizer_default_adamw():
    param = torch.nn.Parameter(torch.ones(2))
    opt = build_optimizer([{"params": [param]}], {"OPTIMIZER_TYPE": "AdamW", "LR": 1e-4, "WEIGHT_DECAY": 1e-3})

    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_schedulefree_has_train_eval_modes():
    pytest_schedulefree = __import__("schedulefree")
    param = torch.nn.Parameter(torch.ones(2))
    opt = build_optimizer(
        [{"params": [param]}],
        {
            "OPTIMIZER_TYPE": "AdamWScheduleFree",
            "LR": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "SCHEDULEFREE_LR": 2.5e-4,
            "SCHEDULEFREE_WEIGHT_DECAY": 1.25e-4,
        },
    )

    assert isinstance(opt, pytest_schedulefree.AdamWScheduleFree)
    set_optimizer_mode(opt, "train")
    set_optimizer_mode(opt, "eval")


def test_build_optimizer_muon_splits_matrix_and_aux_params():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.ones(2))
    opt = build_optimizer(
        [{"params": [matrix, bias]}],
        {
            "OPTIMIZER_TYPE": "AutoMuonWithAuxAdam",
            "LR": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "MUON_LR": 0.005,
            "MUON_WEIGHT_DECAY": 0.01,
            "MUON_ADAM_LR": 0.00025,
            "MUON_ADAM_WEIGHT_DECAY": 0.01,
        },
    )

    assert isinstance(opt, AutoMuonWithAuxAdam)
    assert [group["use_muon"] for group in opt.param_groups] == [True, False]
