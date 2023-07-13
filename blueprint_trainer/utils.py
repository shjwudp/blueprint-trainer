import os

import torch


def are_the_models_the_same(model_lhs, model_rhs):
    return str(model_lhs.state_dict()) == str(model_rhs.state_dict())

def are_the_optimizers_the_same(optim_lhs, optim_rhs):
    return str(optim_lhs.state_dict()) == str(optim_rhs.state_dict())

def are_the_lr_scheduler_the_same(scheduler_lhs, scheduler_rhs):
    return str(scheduler_lhs.state_dict()) == str(scheduler_rhs.state_dict())

def get_save_path(ckpt_dir, step):
    return os.path.join(ckpt_dir, f"ckpt-{step}")

def save_checkpoint(ckpt_dir, step, model, optimizer, lr_scheduler):
    save_path = get_save_path(ckpt_dir, step)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }, save_path)

def load_checkpoint(ckpt_dir, step, model, optimizer, lr_scheduler):
    save_path = get_save_path(ckpt_dir, step)
    ckpt = torch.load(save_path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    return model, optimizer, lr_scheduler

def delete_checkpoint(ckpt_dir, step):
    save_path = get_save_path(ckpt_dir, step)
    os.remove(save_path)
