import argparse

import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl
from S5.dataloaders.synthetics import ICLDataModule

from hyena import Hyena


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint_filepath", required=True)
    args = parser.parse_args()
    
    return args


def get_icl_synthetics_dataset(cfg):
    assert cfg.type == "icl_synthetics"
    dataset_cfg = OmegaConf.to_container(cfg, resolve=True)
    del dataset_cfg["type"]
    dataset_obj = ICLDataModule(**dataset_cfg)
    dataset_obj.setup()

    return dataset_obj


def main():
    args = get_args()
    
    global blueprint

    OmegaConf.register_new_resolver("eval", eval)
    blueprint = OmegaConf.load(args.blueprint_filepath)

    hyena = Hyena(lr=blueprint.learning_rate.base, **blueprint.model.config)
    dataset_obj = get_icl_synthetics_dataset(blueprint.dataset)

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=hyena, train_dataloaders=dataset_obj.train_dataloader())


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
