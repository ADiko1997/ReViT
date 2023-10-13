import math
import os
from datetime import datetime
import torch
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from torch import nn
import numpy as np

def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    Return:
        None
    Raises:
        RuntimeError
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))



def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
        ignore_bn (bool): ignore batch_norm layer paramteres
    Return:
        count (int): number of parameters
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count



def _get_model_analysis_input(cfg):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.
    Returns:
        inputs: the input for model analysis.
    Raises:
        None
    """
    rgb_dimension = 3
    input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.crop_size,
            cfg.DATA.crop_size,
        )

    model_inputs = input_tensors.unsqueeze(0)
    if cfg.DEVICE.num_gpu:
        model_inputs = model_inputs.cuda(non_blocking=True)

    inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
    Returns:
        float: the total number of count of the given model.
    Raises:
        None
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg)
    count_dict, unsupported = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count