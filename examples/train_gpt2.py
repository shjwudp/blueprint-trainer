from dataset_utils import prepare_wikitext_dataset
from example_utils import optimizer_constructor, lr_scheduler_constructor

import time
import os
import functools
import argparse

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, default_data_collator
from datasets import load_dataset

from blueprint_trainer import Trainer


device = "cuda"


def get_gpt2_and_tokenizer(model_path):
    config = GPT2Config.from_pretrained(model_path)
    gpt2 = GPT2LMHeadModel(config).to(device=device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return gpt2, tokenizer


def model_forward(model, batch):
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    output = model(input_ids=ids, labels=labels, return_dict=True)
    loss = output.loss

    return loss, {}


def return_model_eval(eval_dataset):
    dl = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=1, generator=torch.Generator(device=device),
    )
    def model_eval(model):
        model.eval()
        eval_start_timestamp = time.time()
        losses = []
        for batch in dl:
            with torch.no_grad():
                loss, _ = model_forward(model, batch)
            losses.append(loss.reshape(1))
        eval_loss = torch.cat(losses).mean()
        model.train()
        
        return {"eval_loss": eval_loss}, time.time() - eval_start_timestamp

    return model_eval


def my_logging_function(metrics, step=None, commit=True):
    print(f"step-{step}, " + ", ".join([f"{key}: {value}" for key, value in metrics.items()]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint_filepath", required=True)
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    
    global blueprint, device
    torch.set_default_device(device)

    trainer = Trainer(blueprint_filepath=blueprint.blueprint_filepath)
    blueprint = trainer.blueprint

    gpt2, tokenizer = get_gpt2_and_tokenizer(blueprint.model)
    wikitext = load_dataset(blueprint.dataset[0].path, blueprint.dataset[0].name)
    wikitext = prepare_wikitext_dataset(wikitext, tokenizer)
    model_eval_func = return_model_eval(wikitext["validation"])
    train_dataset = wikitext["train"]
    dataloader_kwargs = dict(
        generator=torch.Generator(device=device),
        collate_fn=default_data_collator,
        num_workers=1,
    )

    trainer.prepare(
        model_forward=model_forward,
        model_eval=model_eval_func,
        log=my_logging_function,
        optimizer_constructor=functools.partial(
            optimizer_constructor,
            optim_opt=blueprint.optimizer,
            lr_conf=blueprint.learning_rate,
        ),
        lr_scheduler_constructor=functools.partial(
            lr_scheduler_constructor,
            lr_conf=blueprint.learning_rate,
        ),
        train_dataset=train_dataset,
        dataloader_kwargs=dataloader_kwargs,
    )
    trainer.test_blueprint(gpt2)
    trainer.training_from_scratch(gpt2)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
