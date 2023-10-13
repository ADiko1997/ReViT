
# Credits to  @Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Learning rate policy."""

import math


def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.SOLVER.lr_policy)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.SOLVER.warmup_epochs:
        lr_start = cfg.SOLVER.warmup_start_lr
        lr_end = get_lr_func(cfg.SOLVER.lr_policy)(cfg, cfg.SOLVER.warmup_epochs)
        alpha = (lr_end - lr_start) / cfg.SOLVER.warmup_epochs
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = cfg.SOLVER.warmup_epochs if cfg.SOLVER.cosine_after_warmup else 0.0
    assert cfg.SOLVER.cosine_end_lr < cfg.SOLVER.base_lr
    return (
        cfg.SOLVER.cosine_end_lr
        + (cfg.SOLVER.base_lr - cfg.SOLVER.cosine_end_lr)
        * (
            math.cos(math.pi * (cur_epoch - offset) / (cfg.SOLVER.max_epochs - offset))
            + 1.0
        )
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg: configs. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.lrs[ind] * cfg.SOLVER.base_lr


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg: configs. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.steps + [cfg.SOLVER.max_epochs]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]