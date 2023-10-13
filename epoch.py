from cmath import isnan
import math
import torch
import numpy as np
import dataset.data as data
import utils.distributed as du
import utils.metrics as metrics
import model.optimizer as optim
import model.loss as losses
from dataset.mixup import MixUp
import tqdm
import torch.distributed as dist


def train_epoch(
    train_loader,
    model,
    solver,
    cur_epoch,
    scaler,
    rank,
    cfg,
    sampler,
    writer=None
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        solver (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs.
    """
    model.train()
    data_size = len(train_loader)
    running_loss = 0
    running_top_1_correct = 0 
    running_top_5_correct = 0
    loss_fun = losses.get_loss_func(cfg.MODEL.loss_func)


    if cfg.MIXUP.enable:
        mixup_fn = MixUp(
            mixup_alpha = cfg.MIXUP.mixup_alpha,
            cutmix_alpha = cfg.MIXUP.cutmix_alpha,
            mix_prob = cfg.MIXUP.mixup_prob,
            switch_prob = cfg.MIXUP.switch_prob,
            label_smoothing = cfg.MIXUP.labels_smooth_value,
            num_classes = cfg.MODEL.num_classes
        )
        # print("mixup works")
    if cfg.DEVICE.num_gpu > 1:
        print(f"rank {rank} starting epoch {cur_epoch}")
        sampler.set_epoch(cur_epoch)
    
    # lr = optim.get_epoch_lr(cur_epoch=cur_epoch, cfg=cfg) #Initial lr
    # print(f"lr: {lr}")
    # optim.set_lr(solver, new_lr=lr)
    
    for cur_iter, (inputs, labels) in enumerate(tqdm.tqdm(train_loader)):
        if cfg.DEVICE.num_gpu:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()

        if cfg.MIXUP.enable:
            inputs, labels = mixup_fn(inputs, labels)

        lr = optim.get_epoch_lr(cur_epoch=cur_epoch+float(cur_iter)/data_size, cfg=cfg)
        # lr = 0.0000005
        optim.set_lr(solver, new_lr=lr)

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.mixed_precision):
            preds = model(inputs)
        loss = loss_fun(preds, labels)
        scaled_loss = loss/cfg.SOLVER.accumulate_steps

        if math.isnan(loss):
            raise RuntimeError("ERROR: Got NaN losses")

        if cfg.SOLVER.accumulate:

            scaler.scale(scaled_loss).backward()

            if (cur_iter+1) % cfg.SOLVER.accumulate_steps==0:     

                # print(f"Accumulating on iter: {cur_iter+1}")   
                scaler.unscale_(solver)

                if cfg.SOLVER.clip_grad_val:
                    torch.nn.utils.clip_grad_value_(
                        parameters=model.parameters(), 
                        clip_value = cfg.SOLVER.clip_grad_val
                    )
                
                elif cfg.SOLVER.clip_grad_l2norm:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(), 
                        max_norm=cfg.SOLVER.clip_grad_l2norm
                    )
                
        
                scaler.step(solver)
                scaler.update()
                solver.zero_grad()
                # lr = optim.get_epoch_lr(cur_epoch=cur_epoch+float(cur_iter)/(data_size/cfg.SOLVER.accumulate_steps), cfg=cfg)
                # # lr = 0.0000005
                # optim.set_lr(solver, new_lr=lr)
        else:
            scaler.scale(loss).backward()
            if cfg.SOLVER.clip_grad_val:
                torch.nn.utils.clip_grad_value_(
                    parameters=model.parameters(), 
                    clip_value=cfg.SOLVER.clip_grad_val
                )
            
            elif cfg.SOLVER.clip_grad_l2norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), 
                    max_norm=cfg.SOLVER.clip_grad_l2norm
                )
            
            scaler.step(solver)
            scaler.update()
            solver.zero_grad()
            lr = optim.get_epoch_lr(cur_epoch=cur_epoch+float(cur_iter)/data_size, cfg=cfg)
            # lr = 0.0000005
            optim.set_lr(solver, new_lr=lr)


        if cfg.MIXUP.enable:
            #Uniting the probabilities of top2 predictions (soft labels) into 1 to calculate accuracy
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        num_topks_correct = metrics.topk_correct(preds, labels, (1, 5))
        top_1_correct = num_topks_correct[0]
        top_5_correct = num_topks_correct[1]
        # top_1_acc, top_5_acc = [(x/preds.shape[0]) for x in num_topks_correct]

        if cfg.DEVICE.num_gpu > 1:
            loss, top_1_correct, top_5_correct = du.all_reduce(tensors=[loss, top_1_correct, top_5_correct])
            # loss = dist.all_reduce(torch.tensor(loss).cuda(), op=dist.ReduceOp.SUM)
            # top_1_correct = dist.all_reduce(torch.tensor(top_1_correct).cuda(), op=dist.ReduceOp.SUM)
            # top_5_correct = dist.all_reduce(torch.tensor(top_5_correct).cuda(), op=dist.ReduceOp.SUM)

        
        running_loss += loss.item()
        running_top_1_correct += top_1_correct.item()
        running_top_5_correct += top_5_correct.item()
    
    if cfg.DEVICE.num_gpu > 1:
        if rank == 0:
            print(f"\Training Epoch {cur_epoch} -> loss: {(running_loss/train_loader.__len__())/cfg.DEVICE.num_gpu} \
                top 1 accuracy: {running_top_1_correct/train_loader.dataset.__len__()} \
                top 5 accuracy: {running_top_5_correct/train_loader.dataset.__len__()}\
                lr: {lr}\
            ")

            if writer !=None:

                writer.add_scalar(
                    'Train Top1 accuracy',
                    running_top_1_correct/train_loader.dataset.__len__(),
                    cur_epoch + 1
                )

                writer.add_scalar(
                    'lr',
                    lr,
                    cur_epoch + 1
                )

                writer.add_scalar(
                    'Train loss',
                    (running_loss/train_loader.__len__())/cfg.DEVICE.num_gpu,
                    cur_epoch + 1
                )

    else:
        print(f"\
            Training Epoch {cur_epoch} -> loss: {running_loss/train_loader.__len__()} \
            top 1 accuracy: {running_top_1_correct/train_loader.dataset.__len__()} \
            top 5 accuracy: {running_top_5_correct/train_loader.dataset.__len__()}\
            lr: {lr}\
        ")

        if writer !=None:

            writer.add_scalar(
                'Train Top1 accuracy',
                running_top_1_correct/train_loader.dataset.__len__(),
                cur_epoch + 1
            )

            writer.add_scalar(
                'lr',
                lr,
                cur_epoch + 1
            )

            writer.add_scalar(
                'Train loss',
                running_loss/train_loader.__len__(),
                cur_epoch + 1
            )


        
    return running_loss, running_top_1_correct, running_top_5_correct


def val_epoch(
    val_loader,
    model,
    cur_epoch,
    rank,
    cfg,
    solver,
    best_model=0,
    writer=None
):
    data_size = len(val_loader)
    running_loss = 0
    running_top_1_correct = 0 
    running_top_5_correct = 0
    model.eval()

    with torch.no_grad():

        model.training=False
        for cur_iter, (inputs, labels) in enumerate(tqdm.tqdm(val_loader)):
            if cfg.DEVICE.num_gpu:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()

            # with torch.cuda.amp.autocast(enabled=cfg.TRAIN.mixed_precision):
            preds = model(inputs)

            num_topks_correct = metrics.topk_correct(preds, labels, (1, 5))
            top_1_correct = num_topks_correct[0]
            top_5_correct = num_topks_correct[1]

            # top_k_accuracies = metrics.topk_accuracies(preds, labels, (1, 5))
            # top_1_acc = top_k_accuracies[0]
            # top_5_acc = top_k_accuracies[1]

            if cfg.DEVICE.num_gpu > 1:
                top_1_correct, top_5_correct = du.all_reduce([top_1_correct, top_5_correct], average=False)
                # top_1_acc, top_5_acc = du.all_reduce([top_1_acc, top_5_acc], average=True)
            
            running_top_1_correct += top_1_correct.item()
            running_top_5_correct += top_5_correct.item()
            # top_1_acc = top_1_acc.item()
            # top_5_acc = top_5_acc.item()
    
        if cfg.DEVICE.num_gpu > 1:
            if rank == 0:
                print(f"Validation top 1 accuracy: {running_top_1_correct/val_loader.dataset.__len__()} \
                    top 5 accuracy: {running_top_5_correct/val_loader.dataset.__len__()}\
                ")

                if writer !=None:

                    writer.add_scalar(
                        'Val Top1 accuracy',
                        running_top_1_correct/val_loader.dataset.__len__(),
                        cur_epoch + 1
                    )

                # writer.add_scalar(
                #     'Val Top5 accuracy',
                #     running_top_5_correct/val_loader.dataset.__len__(),
                #     cur_epoch + 1
                # )

        else:
            print(f"Validation top 1 accuracy: {running_top_1_correct/val_loader.dataset.__len__()} \
                top 5 accuracy: {running_top_5_correct/val_loader.dataset.__len__()}\
            ")

            if writer !=None:

                writer.add_scalar(
                    'Val Top1 accuracy',
                    running_top_1_correct/val_loader.dataset.__len__(),
                    cur_epoch + 1
                )

                # writer.add_scalar(
                #     'Val Top5 accuracy',
                #     running_top_5_correct/val_loader.dataset.__len__(),
                #     cur_epoch + 1
                # )


        if cfg.SOLVER.save and cfg.TRAIN.enable:
            if running_top_1_correct/val_loader.dataset.__len__() >= best_model:
                torch.save(
                                {
                                'epoch': cur_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': solver.state_dict(),
                                }, cfg.SOLVER.checkpoint_path
                            )
            best_model = max(best_model, running_top_1_correct/val_loader.dataset.__len__())

    return running_top_1_correct, running_top_5_correct, best_model