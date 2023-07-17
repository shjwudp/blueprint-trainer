import os
import contextlib

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

def seconds_to_human_friendly_time_str(secs):
    time = secs

    # convert seconds to day, hour, minutes and seconds
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time

    if day:
        time_str = f"{day} day{'s' if day > 1 else ''}"
        if hour:
            time_str += f" and {hour} hour{'s' if hour > 1 else ''}"

        return time_str
    
    if hour:
        time_str = f"{hour} hour{'s' if hour > 1 else ''}"
        if minutes:
            time_str += f" and {minutes} minute{'s' if minutes > 1 else ''}"

        return time_str
    
    if minutes:
        time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
        if minutes:
            time_str += f" and {seconds} second{'s' if seconds > 1 else ''}"

        return time_str
    
    time_str = f"{seconds} second{'s' if seconds > 1 else ''}"

    return time_str

class GradientAccumulator:

    def __init__(self, n_gradient_accumulation_step) -> None:
        self.num_forward = 0
        self.sync_gradients = False
        self.n_gradient_accumulation_step = n_gradient_accumulation_step

    @contextlib.contextmanager
    def accumulate(self, model_no_sync):
        self.num_forward += 1
        self.sync_gradients = self.num_forward % self.n_gradient_accumulation_step == 0
        if self.sync_gradients:
            context = contextlib.nullcontext
        else:
            context = model_no_sync

        with context():
            yield
