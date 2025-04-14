import torch
import os, gc
import argparse

def print_rank0(text: str):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(text)
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            print(text)
    return None


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print_rank0(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_max_memory_allocated()