from dis import dis
import torch
import torch.distributed as dist

def all_reduce(tensors, op=dist.ReduceOp.SUM, average=False):
    """
    All reduce the provided tensors from all machines (GPUs)
    Args:
        tensors (list): tensors produced by machines part of dist processing
        average (bool): decides it the reduction will be trhough averaging or not

    Returns:
        tensor (tensor): reduced tensor 

    Raises:
        None
    """
    for tensor in tensors:
        dist.all_reduce(tensor=tensor, op=op, async_op=False)
    if average:
        world_size = dist.get_world_size() #gets the number of machines
        for tensor in tensors:
            tensor.mul_(1.0/world_size)
    return tensors

def is_master_proc(num_gpus=2):
    """
    Determines if the current process is the master or not
    Master process is the one who initiates the job and where results are gatheres
    Args:
        num_gpus (int): number of gpus used for running the programm
    Returns:
        bool: True if the process is master (rank==0) False otherwise
    Raises:
        None
    """
    if dist.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True
    

def get_world_size():
    """
    Get the number of active machines a.k.a size of the world
    Args:
        None
    Return:
        (int): number of machines
    Raises:
        None
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

