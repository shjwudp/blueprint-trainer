"""
pip install 
"""

import time
import functools
import argparse

import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
import transformers
from einops import rearrange

from blueprint_trainer import Trainer

from safari.models.sequence.simple_lm import SimpleLMHeadModel
from S5.dataloaders.synthetics import ICLDataModule


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


def get_hyena(cfg):
    assert cfg.type == "SimpleLMHeadModel"
    model_cfg = OmegaConf.to_container(cfg.config, resolve=True)

    hyena = SimpleLMHeadModel(**model_cfg).to(device=blueprint.device)
    return hyena


def model_forward(model, batch):
    inputs = batch[0]
    targets = batch[1]

    output, _ = model(input_ids=inputs)
    logits = output.logits

    # logits: [b, s, v] -> [v, b*s]
    logits = rearrange(logits, "b s v -> (b s) v")

    # targets: [b, s] -> [b*s]
    targets = rearrange(targets, "b s -> (b s)")

    loss = F.cross_entropy(logits, targets, ignore_index=-100)

    return loss, {}


def return_model_eval(eval_dataset):
    dl = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True,
        batch_size=1, generator=torch.Generator(device=blueprint.device),
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
    
    global blueprint

    OmegaConf.register_new_resolver("eval", eval)
    trainer = Trainer(blueprint_filepath=args.blueprint_filepath)
    blueprint = trainer.blueprint

    torch.set_default_device(blueprint.device)

    hyena = get_hyena(blueprint.model)

    assert blueprint.dataset.type == "icl_synthetics"
    dataset_cfg = OmegaConf.to_container(blueprint.dataset, resolve=True)
    del dataset_cfg["type"]
    dataset_obj = ICLDataModule(**dataset_cfg)
    dataset_obj.setup()

    model_eval_func = return_model_eval(dataset_obj.dataset["test"])
    train_dataset = dataset_obj.dataset["train"]
    dataloader_kwargs = dict(
        generator=torch.Generator(device=blueprint.device),
        num_workers=10,
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
    trainer.print_blueprint(hyena)
    trainer.training_from_scratch(hyena)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
