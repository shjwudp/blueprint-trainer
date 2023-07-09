from collections import namedtuple
import time
import yaml

import torch
from torch.utils.data import DataLoader, Subset


Blueprint = namedtuple(
    "Blueprint",
    [
        "model",
        "optimizer",
        "learning_rate",
        "dataset",
        "batch_size_plan",
        "checkpoint",
        "evaluation",
    ]
)

LearningRateConfig = namedtuple(
    "LearningRateConfig", ["base", "scheduler", "num_warmup_steps"]
)

DatasetConfigItem = namedtuple(
    "DatasetConfigItem", ["path", "name"]
)

BatchSizePlanItem = namedtuple(
    "BatchSizePlanItem", ["batch_size", "training_nsteps"]
)

CheckpointConfig = namedtuple(
    "CheckpointConfig", ["interval_by_step", "interval_by_time"]
)

EvaluationConfig = namedtuple(
    "EvaluationConfig", ["interval_by_step", "interval_by_time"]
)


class Trainer:
    def __init__(
        self,
        blueprint_json=None,
        blueprint_text=None,
        blueprint_filepath=None,
    ) -> None:
        if blueprint_json:
            pass
        elif blueprint_text:
            blueprint_json = yaml.load(blueprint_text, Loader=yaml.Loader)
        elif blueprint_filepath:
            blueprint_json = yaml.load(open(blueprint_filepath).read(), Loader=yaml.Loader)

        blueprint_json = blueprint_json["blueprint"]
        self.blueprint = Blueprint(
            model=blueprint_json["model"],
            optimizer=blueprint_json["optimizer"],
            learning_rate=LearningRateConfig(
                base=float(blueprint_json["learning_rate"]["base"]),
                scheduler=blueprint_json["learning_rate"]["scheduler"],
                num_warmup_steps=blueprint_json["learning_rate"]["num_warmup_steps"],
            ),
            dataset=[
                DatasetConfigItem(path=x["path"], name=x["name"])
                for x in blueprint_json["dataset"]
            ],
            batch_size_plan=[
                BatchSizePlanItem(
                    batch_size=x["batch_size"],
                    training_nsteps=x["training_nsteps"]
                ) for x in blueprint_json["batch_size_plan"]
            ],
            checkpoint=CheckpointConfig(
                interval_by_step=blueprint_json["checkpoint"]["interval_by_step"],
                interval_by_time=blueprint_json["checkpoint"]["interval_by_time"],
            ),
            evaluation=EvaluationConfig(
                interval_by_step=blueprint_json["evaluation"]["interval_by_step"],
                interval_by_time=blueprint_json["evaluation"]["interval_by_time"],
            )
        )

        self.ready_to_train = False

    def prepare(
        self,
        model_forward=None,
        model_eval=None,
        save_checkpoint=None,
        log=None,
        optimizer=None,
        train_dataset=None,
        dataloader_kwargs=None,
        lr_scheduler=None,
    ):
        self.model_forward = model_forward
        self.model_eval = model_eval
        self.save_checkpoint = save_checkpoint
        self.log = log
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.dataloader_kwargs = dataloader_kwargs
        self.lr_scheduler = lr_scheduler

        # data loader initialization
        n_dataloader = []
        blueprint = self.blueprint
        for plan in blueprint["batch_size_plan"]:
            bs, train_step = plan["batch_size"], plan["training_nsteps"]
            if train_step == -1 or train_step > len(train_dataset)//bs:
                dataloder = DataLoader(
                    train_dataset,
                    batch_size=bs,
                    **dataloader_kwargs,
                )
                n_dataloader.append(dataloder)
                break

            stage_dataset = Subset(train_dataset, torch.arange(bs*train_step))
            train_dataset = Subset(train_dataset, torch.arange(bs*train_step, len(train_dataset)))

            dataloder = DataLoader(
                stage_dataset,
                batch_size=bs,
                **dataloader_kwargs,
            )
            n_dataloader.append(dataloder)
        self.n_dataloader = n_dataloader

    def print_blueprint(self):
        # TODO: Many blueprint details need to be improved, 
        # including time estimation, data consumption-time curve
        pass 

    def training_from_scratch(self, model):
        optimizer = self.optimizer
        n_dataloader = self.n_dataloader
        model_forward = model_forward
        log = self.log
        blueprint = self.blueprint
        model_eval = self.model_eval
        save_checkpoint = self.save_checkpoint
        lr_scheduler = self.lr_scheduler
        eval_interval = blueprint.evaluation.interval_by_step
        checkpoint_interval = blueprint.checkpoint.interval_by_step

        # model training
        completed_steps = 0
        optimizer.zero_grad()
        time_stone = time.time()
        for dl in n_dataloader:
            for batch in dl:
                loss, metrics = model_forward(model, batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                completed_steps += 1

                # logging
                spend_time = time.time() - time_stone
                time_stone = time.time()
                metrics.update({"loss": loss, "spend_time": spend_time})
                log(metrics, step=completed_steps)

                if completed_steps % eval_interval == 0:
                    eval_metrics, spend_time = model_eval(model)
                    log(eval_metrics, commit=False)

                if completed_steps % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, lr_scheduler)
