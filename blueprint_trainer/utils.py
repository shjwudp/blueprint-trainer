import os

import torch


def are_the_models_the_same(model_lhs, model_rhs):
    return str(model_lhs.state_dict()) == str(model_rhs.state_dict())

def are_the_optimizers_the_same(optim_lhs, optim_rhs):
    return str(optim_lhs.state_dict()) == str(optim_rhs.state_dict())

def are_the_lr_scheduler_the_same(scheduler_lhs, scheduler_rhs):
    return str(scheduler_lhs.state_dict()) == str(scheduler_rhs.state_dict())

def get_save_path(dir, step):
    return os.path.join(dir, f"ckpt-{step}")

def save_checkpoint(dir, step, model, optimizer, lr_scheduler):
    save_path = get_save_path(dir, step)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }, save_path)

def load_checkpoint(dir, step, model):
    save_path = get_save_path(dir, step)
    checkpoint = torch.load(save_path)
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"], checkpoint["lr_scheduler_state_dict"]

def delete_checkpoint(dir, step):
    save_path = get_save_path(dir, step)
    os.remove(save_path)
