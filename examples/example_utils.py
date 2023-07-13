import torch
import transformers


def optimizer_constructor(model, optim_opt, lr_conf):
    if optim_opt == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_conf.base)

    return optimizer


def lr_scheduler_constructor(optimizer, lr_conf):
    lr_scheduler = transformers.get_scheduler(
        lr_conf.scheduler,
        optimizer,
        num_warmup_steps=lr_conf.num_warmup_steps,
    )

    return lr_scheduler
