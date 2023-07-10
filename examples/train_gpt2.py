from dataset_utils import prepare_wikitext_dataset

import time
import os

import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, default_data_collator
from datasets import load_dataset

from blueprint_trainer import Trainer


blueprint = """
What It Is:
  GPT2 training blueprint

blueprint:
  model: "gpt2"
  optimizer: "AdamW"
  learning_rate:
    base: 1e-4
    scheduler: "inverse_sqrt"
    num_warmup_steps: 1000

  dataset:
  - path: "wikitext"
    name: "wikitext-103-v1"

  batch_size_plan:
  - batch_size: 2
    training_nsteps: 100
  - batch_size: 4
    training_nsteps: 100
  - batch_size: 8
    training_nsteps: 200
  - batch_size: 16
    training_nsteps: 200
  - batch_size: 32
    training_nsteps: -1

  logging:
    path: "./gpt2_training_log"
    interval_by_step: 1
    interval_by_time: "1h"

  checkpoint:
    path: "./gpt2_checkpoints"
    interval_by_step: 1000
    interval_by_time: "1h"

  evaluation:
    interval_by_step: 1000
    interval_by_time: "1h"
"""

device = "cpu"


def get_gpt2_and_tokenizer(model_path):
    config = GPT2Config.from_pretrained(model_path)
    gpt2 = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return gpt2, tokenizer


def model_forward(model, batch):
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    print(ids.shape)
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
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                loss = model_forward(model, ids)
            losses.append(loss.reshape(1))
        eval_loss = torch.cat(losses).mean()
        model.train()
        
        return {"eval_loss": eval_loss}, time.time() - eval_start_timestamp

    return model_eval


def save_checkpoint(model, optimizer, lr_scheduler):
    pass


def log(metrics, step=None, commit=True):
    pass


def get_optimizer(model, optim_opt, lr_conf):
    if optim_opt == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_conf.base)
        return optimizer
    

def get_lr_scheduler(optimizer, lr_conf):
    return transformers.get_scheduler(
        lr_conf.scheduler,
        optimizer,
        num_warmup_steps=lr_conf.num_warmup_steps,
    )


trainer = Trainer(blueprint_text=blueprint)
blueprint = trainer.blueprint

gpt2, tokenizer = get_gpt2_and_tokenizer(blueprint.model)
wikitext = load_dataset(blueprint.dataset[0].path, blueprint.dataset[0].name)
wikitext = prepare_wikitext_dataset(wikitext, tokenizer)
model_eval_func = return_model_eval(wikitext["validation"])
optimizer = get_optimizer(gpt2, blueprint.optimizer, blueprint.learning_rate)
lr_scheduler = get_lr_scheduler(optimizer, blueprint.learning_rate)
train_dataset = wikitext["train"]
dataloader_kwargs = dict(
    generator=torch.Generator(device=device),
    collate_fn=default_data_collator,
    num_workers=os.cpu_count(),
)

trainer.prepare(
    model_forward=model_forward,
    model_eval=model_eval_func,
    save_checkpoint=save_checkpoint,
    log=log,
    optimizer=optimizer,
    train_dataset=train_dataset,
    dataloader_kwargs=dataloader_kwargs,
    lr_scheduler=lr_scheduler,
)
trainer.test_blueprint(gpt2)
trainer.training_from_scratch(gpt2)
