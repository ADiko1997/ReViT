from ast import Str
from cmath import isnan
import math
from statistics import mode
import torch
import numpy as np
from zmq import device
import dataset.data as data
import utils.distributed as du
from epoch import train_epoch, val_epoch
import yaml
from munch import DefaultMunch
import model.optimizer as optim


def _train(model, cfg, solver, train_loader, val_loader=None, cur_epoch=0, scaler=None, rank=None, writer=None, sampler=None):

    best_model=0
    for epoch in range(cur_epoch, cfg.SOLVER.max_epochs, 1):

        running_loss_tr, running_top_1_acc_tr, running_top_5_acc_tr = train_epoch(
            train_loader=train_loader,
            model=model,
            cur_epoch=epoch,
            scaler=scaler,
            rank=rank,
            cfg=cfg,
            solver=solver,
            writer=writer,
            sampler=sampler
        )

        if epoch % cfg.DEVICE.log_period == 0 and val_loader:
            print(f"Epoch {epoch}")
            running_top_1_acc_val, running_top_5_acc_val, best_model = val_epoch(
            val_loader=val_loader,
            model=model,
            cur_epoch=epoch,
            rank=rank,
            cfg=cfg,
            best_model=best_model,
            solver=solver,
            writer=writer
            )
            # best_model = max(best_model, running_top_1_acc_val)




def _val(model, cfg, val_loader, cur_epoch, solver=None, rank=None, writer=None):
    print(f"Epoch {cur_epoch}")
    running_loss, running_top_1_correct, running_top_5_correct = val_epoch(
            val_loader=val_loader,
            model=model,
            cur_epoch=cur_epoch,
            rank=rank,
            cfg=cfg,
            solver=solver,
            writer=writer
    )
    return running_loss, running_top_1_correct, running_top_5_correct

