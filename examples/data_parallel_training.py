from dataset_utils import prepare_wikitext_dataset

import argparse
from itertools import chain
import time
import os
from contextlib import contextmanager
import contextlib
import functools

import torch
import transformers
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    default_data_collator,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from blueprint_trainer import Trainer


def get_gpt2_and_tokenizer(model_path, device="cpu"):
    config = GPT2Config.from_pretrained(model_path)
    gpt2 = GPT2LMHeadModel(config).to(device=device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return gpt2, tokenizer


def model_forward(model, batch, device):
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    output = model(input_ids=ids, labels=labels, return_dict=True)
    loss = output.loss

    return loss, {}


def model_eval(
    model,
    eval_dataset,
    batch_size=1,
    device="cpu",
):
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=batch_size,
        generator=torch.Generator(device=device),
    )

    model.eval()
    eval_start_timestamp = time.time()
    losses = []
    for batch in eval_dataloader:
        with torch.no_grad():
            loss, _ = model_forward(model, batch, device)
        losses.append(loss.reshape(1))
    eval_loss = torch.cat(losses).mean()
    model.train()
    
    return {"eval_loss": eval_loss}, time.time() - eval_start_timestamp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint_filepath", required=True)
    args = parser.parse_args()
    
    return args


class DataParallel:
    def __init__(self):
        # TODO: init process group with backend "nccl|gloo" does not work, fix this issue. 
        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.sync_gradients = False
        self.completed_steps = 0
        self.num_forward = 0
        self.device = f"cuda:{self.local_rank}"
        self.is_main_process = (self.rank == 0)
        
    @contextmanager
    def main_process_first(self):
        if not self.is_main_process:
            dist.barrier()

        yield

        if self.is_main_process:
            dist.barrier()
            
    def barrier(self):
        dist.barrier()


def print_rank0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args, flush=True)


def my_logging_function(metrics, step=None, commit=True):
    if step:
        print_rank0(f"step-{step}, " + ", ".join([f"{key}: {value}" for key, value in metrics.items()]))


def get_optimizer_and_lr_scheduler(model, optim_opt, lr_conf):
    if optim_opt == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_conf.base)

    lr_scheduler = transformers.get_scheduler(
        lr_conf.scheduler,
        optimizer,
        num_warmup_steps=lr_conf.num_warmup_steps,
    )

    return optimizer, lr_scheduler


def main():
    args = get_args()

    trainer = Trainer(blueprint_filepath=args.blueprint_filepath)
    blueprint = trainer.blueprint

    dp = DataParallel()
    torch.set_default_device(dp.device)

    model, tokenizer = get_gpt2_and_tokenizer(blueprint.model, device=dp.device)
    ddp_model = DistributedDataParallel(model)

    with dp.main_process_first():
        wikitext = load_dataset(blueprint.dataset[0].path, blueprint.dataset[0].name)
        wikitext = prepare_wikitext_dataset(wikitext, tokenizer)

    def moel_eval_func(model):
        eval_func = functools.partial(
            model_eval,
            eval_dataset=wikitext["validation"],
            batch_size=1,
            device=dp.device,
        )

        if dp.is_main_process:
            eval_func(model)
        dp.barrier()

    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
        model,
        blueprint.optimizer,
        blueprint.learning_rate,
    )

    trainer.prepare(
        model_forward=functools.partial(model_forward, device=dp.device),
        model_eval=moel_eval_func,
        log=my_logging_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataset=wikitext["train"],
        dataloader_kwargs=dict(
            collate_fn=default_data_collator,
            generator=torch.Generator(device=dp.device),
            num_workers=os.cpu_count(),
        ),
        dp_handler=dp,
    )

    trainer.test_blueprint(ddp_model)
    trainer.training_from_scratch(ddp_model)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
