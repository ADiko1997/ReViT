"""Optimizer. Modified from Meata (c)"""

import json

import model.lr_policy as lr_policy
import torch


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    Args:
        model (model): model to perform stochastic gradient descent
            optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
            learning rate,  momentum, weight_decay, dampening, and etc.
    """

  
    optim_params = get_param_groups(model, cfg)

    if cfg.SOLVER.optimizing_method == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.base_lr,
            momentum=cfg.SOLVER.momentum,
            weight_decay=cfg.SOLVER.weight_decay,
            dampening=cfg.SOLVER.dampening,
            nesterov=cfg.SOLVER.nesterov,
        )
    elif cfg.SOLVER.optimizing_method == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.weight_decay,
        )
    elif cfg.SOLVER.optimizing_method == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.base_lr,
            eps=1e-08,
            weight_decay=cfg.SOLVER.weight_decay,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.optimizing_method)
        )


def get_param_groups(model, cfg):

    bn_parameters = [] #params for batch_norm layers
    non_bn_parameters = [] #non_batch_norm parameters with active gradients
    zero_parameters = []    #zero parameter objects
    no_grad_parameters = [] #non active gradient objects

    skip = {} #layers to be skipped
    if cfg.DEVICE.num_gpu > 1:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
            skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif any(k in name for k in skip) or (
                (len(p.shape) == 1 or name.endswith(".bias"))
                and cfg.SOLVER.zero_wd_1D_param #performs non weight decay in 1D params like bias
            ):
                zero_parameters.append(p)
            else:
                non_bn_parameters.append(p)

    optim_params = [
        {"params": bn_parameters, "weight_decay": 0.0},
        {
            "params": non_bn_parameters,
            "weight_decay": cfg.SOLVER.weight_decay,
        },
        {"params": zero_parameters, "weight_decay": 0.0},
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_parameters
    ) + len(zero_parameters) + len(
        no_grad_parameters
    ), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(list(model.parameters())),
    )

    print(
        "bn {}, non bn {}, zero {} no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )

    return optim_params


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        ld = param_group["layer_decay"] if "layer_decay" in param_group else 1.0
        param_group["lr"] = new_lr * ld