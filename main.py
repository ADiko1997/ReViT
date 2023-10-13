from email import parser
import os
import torch
import numpy as np
import random
import run
import dataset.data as data
import model.revit_model
import model.reswin
import model.swin_old
import model.optimizer as optim
import utils.model_stats as model_stats
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from collections import OrderedDict
import timm

#tuning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import logging
from ray.tune import CLIReporter

#parallel classes
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from munch import DefaultMunch



logging.disable(logging.INFO)
logging.disable(logging.WARNING)

#Reproductability
np.random.seed(42)
random.seed(42)

if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(42)
else:
    torch.cuda.manual_seed(42)

torch.manual_seed(42)


#Setting environment variables for debugging
os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"]="1"
os.environ['NCCL_LL_THRESHOLD']='0'

__MODELS__ = [
    "ReViT", "ReSwin", "ReMViTv2"
]

def build(cfg):
    """
    Helper function to build neural backbone

    Input:
        cfg: configuration dictionary
    
    Returns:
        net: Neural network moduel, nn.Module object
    
    Raises:
        ValueError: Model is not supported
    """
    model_name = cfg.MODEL.name

    if model_name == "ReMViTv2" or model_name =="remvitv2":
        net = model.revit_model.ReViT(cfg)

    elif model_name == "ReViT" or model_name =="ReViT":
        net = model.revit_model.ReViT(cfg)

    elif model_name == "ReSwin" or model_name == "ReSwin":
        net = model.reswin.SwinTransformer(
            img_size=cfg.DATA.crop_size, patch_size=cfg.ReSwin.patch_size, in_chans=3, 
            num_classes=cfg.MODEL.num_classes, embed_dim=cfg.ReSwin.embed_dim, depths=cfg.ReSwin.depths, num_heads=cfg.ReSwin.num_heads, 
            head_dim=None, window_size=cfg.ReSwin.window_size, mlp_ratio=cfg.ReSwin.mlp_ratio, 
            qkv_bias=cfg.ReSwin.qkv_bias, drop_rate=cfg.ReSwin.drop_rate, attn_drop_rate=cfg.ReSwin.attn_drop_rate, 
            drop_path_rate=cfg.ReSwin.drop_path, ape=cfg.ReSwin.ape, patch_norm=cfg.ReSwin.patch_norm
        )
    elif "Hug" in model_name:
        net = timm.create_model("vit_base_patch16_224", pretrained=cfg.TRAIN.enable)
        net.head = model.revit_model.TransformerBasicHead(dim_in=768, num_classes=cfg.MODEL.num_classes)
    else:
        raise ValueError(f"Model name not supported, please inset one of the following: {__MODELS__}")
    print(f"Model {model_name} built successfully")   
    return net



def parallel_setup(rank, world_size, backend):
    
    """
    Setup funcion for distributed data parallel training on single-machine multi GPUs.
    args:
        rank: id of current running node (0 is the master)
        world_size: number of GPUs available for training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1001'

    #initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)





def run_parallel(train_fn, world_size, cfg):

    """ The running function that activates parallellization through multiprocessing"""

    mp.spawn(train_fn,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)




def parallel_train(rank, world_size, cfg):
    """
    Parallel training
    args:
        rank: id of current running node (0 is the master)
        world_size: number of GPUs available for training
        cfg: Configuration object
    """
    os.environ['LOCAL_RANK'] = str(rank)
    parallel_setup(rank, world_size, cfg.DEVICE.dist_backend)
    torch.cuda.set_device(rank)
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = cfg.SOLVER.base_lr * cfg.TRAIN.batch_size * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr =  cfg.SOLVER.warmup_start_lr * cfg.TRAIN.batch_size * dist.get_world_size() / 512.0
    linear_scaled_min_lr =  cfg.SOLVER.cosine_end_lr * cfg.TRAIN.batch_size * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if cfg.SOLVER.accumulate_steps > 1:
        linear_scaled_lr = linear_scaled_lr * cfg.SOLVER.accumulate_steps
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg.SOLVER.accumulate_steps
        linear_scaled_min_lr = linear_scaled_min_lr * cfg.SOLVER.accumulate_steps

    cfg.SOLVER.base_lr = linear_scaled_lr
    cfg.SOLVER.warmup_start_lr = linear_scaled_warmup_lr
    cfg.SOLVER.cosine_end_lr = linear_scaled_min_lr

    # net = model.revit_model.ReViT(cfg=cfg)
    net = build(cfg)
    train_loader, val_loader, sampler = data.pytorch_dataloader(cfg=cfg, batch_size=cfg.TRAIN.batch_size, sampler=True, world_size=world_size, rank=rank)
    scaler = torch.cuda.amp.GradScaler()
    solver = optim.construct_optimizer(model=net, cfg=cfg)

    net = net.to(rank)
    net = DDP(
        net,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )
    if cfg.SOLVER.load:
        mapping = "cuda:{rank}".format(rank=rank)
        checkpoint = torch.load(cfg.SOLVER.load_path, map_location=mapping)
        net.load_state_dict(checkpoint['model_state_dict'])
        solver.load_state_dict(checkpoint['optimizer_state_dict'])
        cfg.SOLVER.start_epoch = checkpoint['epoch'] + 1


    if rank == 0:
        params = model_stats.params_count(net)
        print(net)
        print("Net PArams: ", params)


    
    writer = SummaryWriter(cfg.SOLVER.summary)

    if cfg.TRAIN.enable:
        run._train(
            model=net,
            cfg=cfg,
            solver=solver,
            train_loader=train_loader,
            val_loader=val_loader,
            cur_epoch=cfg.SOLVER.start_epoch,
            scaler=scaler,
            rank=rank,
            writer=writer,
            sampler=sampler
        )

    else:
        run._val(
            model=net,
            cfg=cfg,
            cur_epoch=cfg.SOLVER.start_epoch,
            rank=rank,
            val_loader=val_loader,
            writer=None
        )
    
    return


    

#Epoch 1- 46 no augmentation
# epoch 46 - tbd randomcropping and denoising

def main_(cfg):
    """
    Main Function designed for training,, validation and hyper parameter tunning.
    As input it takes the setup parameters from file config.yaml found in Home_dir/Config directory
    Based on the net.mode parameter decides if it will start aa training or validation session and
    based on opt.fine_tune it decides if it is going through hyper parameter tuning or a full normal session.
    """

    #Declare summary writer
    if cfg.SOLVER.summary:
        writer = SummaryWriter(cfg.SOLVER.summary)
    else:
        writer=None

    #set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Let's use", torch.cuda.device_count(), "GPUs!")    

    #Hyperparameter tunning
    if cfg.SOLVER.fine_tune:

        tune_search_space = {
            "lr": tune.loguniform(1e-3, 1e-5),
            "batch_size": tune.choice([16, 32, 64]),
            "patch_size": tune.choice([4, 8, 16]),
            "blocks_per_stage": [
                tune.choice([1, 2]), tune.choice([ 2, 3, 4, 5]), 
                tune.choice([ 3, 4, 5]),tune.choice([2, 3, 4])  #tuning the number of blocks for each of 4 stages
                ],
        }

        #Wrap the training setup into a function to be used by RayTune for hyper parameter tunning
        def fine_tune(tune_space, checkpoint_dir=None):

            net = build(cfg)
            net.to(device)
            # train_loader, val_loader = pytorch_dataloader(cfg=cfg, batch_size=tune_space['batch_size'])
            train_loader = data.make_data_loader(data_dir=cfg.DATA.path, mode='train', transform=True, batch_size=tune_space['batch_size'], shuffle=True, cfg=cfg)
            val_loader = data.make_data_loader(data_dir=cfg.DATA.path, mode='val', transform=False, batch_size=tune_space['batch_size'], shuffle=False, cfg=cfg)
        
            scaler = torch.cuda.amp.GradScaler()
            solver = optim.construct_optimizer(model=net, cfg=cfg)


            run._train(
                    model=net,
                    cfg=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cur_epoch=0,
                    scaler=scaler
                )

        reporter = CLIReporter(max_report_frequency=540)

        scheduler = ASHAScheduler(
                max_t=10,
                grace_period=2,
                reduction_factor=2
            )

        result = tune.run(
                tune.with_parameters(fine_tune),
                resources_per_trial={"cpu": 12, "gpu": 1},
                config=tune_search_space,
                metric="accuracy",
                mode="max",
                num_samples=50,
                scheduler=scheduler,
                progress_reporter=reporter
            )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))

        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]
            ))

        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
            ))

        return


    else:

        if cfg.SOLVER.dist and cfg.DEVICE.num_gpu>1:
            print("Parallel Running")
            run_parallel(train_fn=parallel_train, world_size=torch.cuda.device_count(), cfg=cfg)
            return
            
        
        elif cfg.TRAIN.dataset != 'tiny-imagent':
            train_loader, val_loader = data.pytorch_dataloader(
                cfg=cfg, 
                batch_size=cfg.TRAIN.batch_size
            )
        
        # elif cfg.TRAIN.dataset == 'cifar10':
        #     train_loader, val_loader = data.pytorch_dataloader(
        #         cfg=cfg, 
        #         batch_size=cfg.TRAIN.batch_size
        #     )

        # elif cfg.TRAIN.dataset == 'cifar100':
        #     train_loader, val_loader = data.pytorch_dataloader(
        #         cfg=cfg, 
        #         batch_size=cfg.TRAIN.batch_size
        #     )

        # elif cfg.TRAIN.dataset == 'pets':
        #     train_loader, val_loader = data.pytorch_dataloader(
        #         cfg=cfg, 
        #         batch_size=cfg.TRAIN.batch_size
        #     )

        else:
            if cfg.TRAIN.enable:
                train_loader = data.make_data_loader(
                    data_dir=cfg.DATA.path,
                    mode='train',
                    transform=True,
                    batch_size=cfg.TRAIN.batch_size,
                    shuffle=True,
                    cfg=cfg,
                    num_workers=12,
                )

            val_loader = data.make_data_loader(
                data_dir=cfg.DATA.path,
                mode='val',
                transform=False,
                batch_size=cfg.TRAIN.batch_size,
                shuffle=False,
                cfg=cfg,
                num_workers=12
            )
        
        #build network
        net = build(cfg)
        net.to(device)
        #create optimizers
        scaler = torch.cuda.amp.GradScaler()
        solver = optim.construct_optimizer(model=net, cfg=cfg)

    # get_model_named_params(net)
    net.to(device)
    if cfg.SOLVER.load:
        mapping = "{rank}".format(rank=device)
        checkpoint = torch.load(cfg.SOLVER.load_path, map_location=mapping)
        model_state_dict = OrderedDict()

        try:
            for k, v in checkpoint['model'].items():
                model_state_dict[k.replace('module.', "")] = v
        except:
            model_state_dict = checkpoint['model_state_dict']

        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in model_state_dict.items() if k in net_dict and v.shape == net_dict[k].shape}
        net.load_state_dict(pretrained_dict, strict=False)
        print("Weights Loaded succesfully!!!")
        if cfg.SOLVER.finetune:
            for name, param in net.named_parameters():
                if "blocks"  in name:
                    print(name)
                    param.requires_grad = False


    #Train/Valid
    if cfg.TRAIN.enable:

        if cfg.SOLVER.load:
            try:
                solver.load_state_dict(checkpoint['optimizer_state_dict'])
                cfg.SOLVER.start_epoch = checkpoint['epoch']+1
            except:
                cfg.SOLVER.start_epoch = 0
        else:
            cfg.SOLVER.start_epoch = 0
        run._train(
            model=net,
            cfg=cfg,
            solver=solver,
            train_loader=train_loader,
            val_loader=val_loader,
            cur_epoch=cfg.SOLVER.start_epoch,
            scaler=scaler,
            rank=None,
            writer=writer
        )
    
    else:
        run._val(
            model=net,
            cfg=cfg,
            cur_epoch=cfg.SOLVER.start_epoch,
            rank=None,
            val_loader=val_loader,
            writer=writer
        )
    
    return



if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    parser.add_argument('--config_path',
                       type=str,
                       help='the path to configuration file')

    parser.add_argument('--num_gpus',
                    type=int,
                    help='the path to configuration file')

    #parse the arguments
    args = parser.parse_args()

    #check if file exists and is yaml
    if os.path.exists(str(args.config_path)) and '.yaml' in str(args.config_path):
        config_path = str(args.config_path)
    else:
        raise ValueError("Path does not exist or is not yaml")
    
    #read config file and turn it into a hieararchichal dict
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream=stream)
    cfg = DefaultMunch.fromDict(cfg)    

    #num of gpus to use
    num_gpus = args.num_gpus
    if num_gpus != None:
        if num_gpus > int(torch.cuda.device_count()):
            print(f"Inserted num_gpus={num_gpus} is bigger than the supported number")
        else:
            cfg.DEVICE.num_gpu = num_gpus
            print("Num GPUS: ", num_gpus)

    #call the program
    main_(cfg)