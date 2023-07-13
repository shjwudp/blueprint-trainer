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


def fixed_seq_length_of_datasets(
    datasets,
    fixed_seq_length,
    tokenizer,
    load_from_cache_file=False,
):
    block_size = fixed_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Padding in front of tokens to align it with the group size.
        if total_length % block_size != 0:
            count_pad_ids = block_size - (total_length % block_size)
            concatenated_examples[list(examples.keys())[0]] = count_pad_ids*[tokenizer.pad_id] + concatenated_examples[list(examples.keys())[0]]

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


def data_sampler(dp, dataset):
    return DistributedSampler(
        dataset,
        dp.world_size,
        dp.rank,
        shuffle=False
    )


def prepare_dataset(
    dataset_name,
    dataset_config_name,
    tokenizer,
    seq_length=512,
    overwrite_cache=False,
    length_expolation_eval=False,
):
    raw_datasets = load_dataset(dataset_name, dataset_config_name)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name], add_eos_token=True),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    lm_datasets = fixed_seq_length_of_datasets(
        tokenized_datasets,
        seq_length,
        tokenizer,
        load_from_cache_file=not overwrite_cache,
    )

    if length_expolation_eval:
        eval_datasets = []
        del tokenized_datasets["train"]
        for eval_seq_length in [seq_length, seq_length*2, seq_length*8]:
            lm_datasets = fixed_seq_length_of_datasets(
                tokenized_datasets,
                eval_seq_length,
                tokenizer,
                load_from_cache_file=not overwrite_cache,
            )
            eval_dataset = lm_datasets["validation"]
            eval_datasets.append(eval_dataset)
    else:
        eval_datasets = [lm_datasets["validation"]]

    return lm_datasets["train"], eval_datasets


def model_forward(model, ids):
    output = model(ids=ids, return_loss=True, return_metrics=True)
    loss = output.loss

    return loss, output.metrics


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
        ids = batch["input_ids"]
        with torch.no_grad():
            loss = model_forward(model, ids)
        losses.append(loss.reshape(1))
    eval_loss = torch.cat(losses).mean()
    model.train()
    
    return {"eval_loss": eval_loss}, time.time() - eval_start_timestamp


def model_eval_with_multiple_dataset(
    model,
    eval_datasets,
    batch_size=1,
    device="cpu"
):
    total_spend_time = 0
    metrics = {}
    for i, d in enumerate(eval_datasets):
        m, spend_time = model_eval(model, d, batch_size, device)
        metrics[f"eval_loss_{i}"] = m["eval_loss"]
        total_spend_time += spend_time

    return metrics, total_spend_time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint_filepath", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--length_expolation_eval", action="store_true")
    args = parser.parse_args()
    
    return args


class DataParallel:
    def __init__(self):
        # TODO: init process group with backend "nccl|gloo" does not work, fix this issue. 
        dist.init_process_group(backend="gloo")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.sync_gradients = False
        self.completed_steps = 0
        self.num_forward = 0
        self.device = f"cpu"
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
    torch.set_default_device(f"cpu")

    print(f"model {blueprint.model} is being created...")
    model, tokenizer = get_gpt2_and_tokenizer(blueprint.model, device=dp.device)
    ddp_model = DistributedDataParallel(model)
    print(f"model {blueprint.model} has been created.")
    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"num_parameters {num_parameters}")

    with dp.main_process_first():
        train_dataset, eval_datasets = prepare_dataset(
            dataset_name=blueprint.dataset.path,
            dataset_config_name=blueprint.dataset.name,
            tokenizer=tokenizer,
            length_expolation_eval=args.length_expolation_eval,
        )

    def moel_eval_func(dataset):
        if not args.length_expolation_eval:
            eval_func = functools.partial(
                model_eval,
                eval_dataset=eval_datasets[0],
                batch_size=args.eval_batch_size,
                device=dp.device,
            )
        else:
            eval_func = functools.partial(
                model_eval_with_multiple_dataset,
                eval_datasets=eval_datasets,
                batch_size=args.eval_batch_size,
                device=dp.device,
            )

        if dp.is_main_process:
            eval_func(dataset)
        dp.barrier()

    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
        model,
        blueprint.optimizer,
        blueprint.learning_rate,
    )

    trainer.prepare(
        model_forward=model_forward,
        model_eval=moel_eval_func,
        log=my_logging_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataset=train_dataset,
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
