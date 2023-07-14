from blueprint_trainer.utils import (
    are_the_models_the_same,
    are_the_optimizers_the_same,
    are_the_lr_scheduler_the_same,
    seconds_to_human_friendly_time_str,
)
import blueprint_trainer.utils as blueprint_utils

from collections import namedtuple
import time
import yaml
import copy

from torch.utils.data.distributed import DistributedSampler
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
    "CheckpointConfig", ["path", "interval_by_step", "interval_by_time"]
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

        j = blueprint_json["blueprint"]
        self.blueprint = Blueprint(
            model=j["model"],
            optimizer=j["optimizer"],
            learning_rate=LearningRateConfig(
                base=float(j["learning_rate"]["base"]),
                scheduler=j["learning_rate"]["scheduler"],
                num_warmup_steps=j["learning_rate"]["num_warmup_steps"],
            ),
            dataset=[
                DatasetConfigItem(path=x["path"], name=x["name"])
                for x in j["dataset"]
            ],
            batch_size_plan=[
                BatchSizePlanItem(
                    batch_size=x["batch_size"],
                    training_nsteps=x["training_nsteps"]
                ) for x in j["batch_size_plan"]
            ],
            checkpoint=CheckpointConfig(
                path=j["checkpoint"]["path"],
                interval_by_step=j["checkpoint"]["interval_by_step"],
                interval_by_time=j["checkpoint"]["interval_by_time"],
            ),
            evaluation=EvaluationConfig(
                interval_by_step=j["evaluation"]["interval_by_step"],
                interval_by_time=j["evaluation"]["interval_by_time"],
            )
        )

        self.blueprint_completed_testing = False

    def prepare(
        self,
        model_forward=None,
        model_eval=None,
        save_checkpoint=None,
        load_checkpoint=None,
        delete_checkpoint=None,
        log=None,
        optimizer_constructor=None,
        lr_scheduler_constructor=None,
        train_dataset=None,
        dataloader_kwargs=None,
        dp_handler=None,
    ):
        self.model_forward = model_forward
        self.model_eval = model_eval
        self.log = log
        self.optimizer_constructor = optimizer_constructor
        self.lr_scheduler_constructor = lr_scheduler_constructor
        self.train_dataset = train_dataset
        self.dataloader_kwargs = dataloader_kwargs
        self.dp_handler = dp_handler

        # data loader initialization
        n_dataloader = []
        blueprint = self.blueprint

        def get_dataloader(dataset, batch_size):
            dl_kwargs = self.dataloader_kwargs
            if dp_handler:
                dp = dp_handler
                # TODO: Supports data parallelism with uneven batch size 
                # distribution, and variable-length data sets have such 
                # usage scenarios
                assert batch_size % dp.world_size == 0
                assert "sampler" not in dataloader_kwargs
                dl_kwargs["sampler"] = DistributedSampler(
                    dataset, dp.world_size, dp.rank, shuffle=False
                )
                batch_size = batch_size//dp.world_size

            dataloader = DataLoader(
                dataset, batch_size=batch_size,
                **dataloader_kwargs,
            )

            return dataloader

        for plan in blueprint.batch_size_plan:
            bs, train_step = plan.batch_size, plan.training_nsteps
            if train_step == -1 or bs*train_step > len(train_dataset):
                dataloader = get_dataloader(train_dataset, bs)
                n_dataloader.append(dataloader)
                break

            stage_dataset = Subset(train_dataset, range(bs*train_step))
            train_dataset = Subset(train_dataset, range(bs*train_step, len(train_dataset)))

            dataloader = get_dataloader(stage_dataset, bs)
            n_dataloader.append(dataloader)
        self.n_dataloader = n_dataloader

        # checkpoint functions check and alert
        if any([save_checkpoint, load_checkpoint, delete_checkpoint]):
            assert all([save_checkpoint, load_checkpoint, delete_checkpoint]), \
                f"Please make sure that all checkpoint functions are set, or none of them are set."
            
            self.save_checkpoint = save_checkpoint
            self.load_checkpoint = load_checkpoint
            self.delete_checkpoint = delete_checkpoint
        else:
            # No function passed in, use the preset checkpoint functions
            self.save_checkpoint = blueprint_utils.save_checkpoint
            self.load_checkpoint = blueprint_utils.load_checkpoint
            self.delete_checkpoint = blueprint_utils.delete_checkpoint

    def print_blueprint(self):
        # TODO: Many blueprint details need to be improved, 
        # including time estimation, data consumption-time curve
        pass

    def memory_stress_test(self, model):
        print("Start Memory Stress Test..")
        optimizer = self.optimizer_constructor(model)
        n_dataloader = self.n_dataloader
        model_forward = self.model_forward
        lr_scheduler = self.lr_scheduler_constructor(optimizer)

        # Memory Stress Test
        completed_steps = 0
        optimizer.zero_grad()
        next_threshold_of_data_amount = 0
        for dl in n_dataloader:
            for batch in dl:
                the_amount_of_data = sum(x.numel() for x in batch.values())

                if the_amount_of_data <= next_threshold_of_data_amount:
                    continue
                next_threshold_of_data_amount = the_amount_of_data

                loss, _ = model_forward(model, batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                completed_steps += 1
        print("Memory Stress Test done.")

    def test_model_evaluation(self, model):
        print("Start Model Evaluation Test..")
        model_eval = self.model_eval

        # Test Model Evaluation
        model_eval(model)
        print("Model Evaluation Test done.")

    def test_checkpoint_save_and_load(self, model):
        print("Start Checkpoint Save & Load Test..")
        completed_steps = 0
        optimizer = self.optimizer_constructor(model)
        lr_scheduler = self.lr_scheduler_constructor(optimizer)

        # Test Checkpoint Save & Load
        if self.save_checkpoint:
            ckpt_dir = self.blueprint.checkpoint.path
            self.save_checkpoint(
                ckpt_dir=ckpt_dir,
                step=completed_steps,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

            saved_model, saved_optimizer, saved_lr_scheduler = \
                self.load_checkpoint(
                    ckpt_dir=ckpt_dir,
                    step=completed_steps,
                    model=copy.deepcopy(model),
                    optimizer=copy.deepcopy(optimizer),
                    lr_scheduler=copy.deepcopy(lr_scheduler),
                )

            assert are_the_models_the_same(model, saved_model)
            assert are_the_optimizers_the_same(optimizer, saved_optimizer)
            assert are_the_lr_scheduler_the_same(lr_scheduler, saved_lr_scheduler)

            self.delete_checkpoint(ckpt_dir=ckpt_dir, step=completed_steps)
        print("Checkpoint Save & Load Test done.")

    def estimate_training_time(self, model):
        print("Start Estimate Training Time..")
        estimated_training_time = 0
        optimizer = self.optimizer_constructor(model)
        n_dataloader = self.n_dataloader
        model_forward = self.model_forward
        lr_scheduler = self.lr_scheduler_constructor(optimizer)

        # Estimate Training Time
        optimizer.zero_grad()
        for dl in n_dataloader:
            batch_count = 0
            sample_count = 0
            sample_time = 0
            for batch in dl:
                batch_count += 1

                if sample_count > 10:
                    continue

                start_timestamp = time.time()
                loss, _ = model_forward(model, batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                sample_time += time.time() - start_timestamp
                sample_count += 1

            if sample_count:
                estimated_training_time += batch_count * (sample_time / sample_count)

        print(f"Estimate Training Time Done. The training is expected to take {seconds_to_human_friendly_time_str(estimated_training_time)}.")

    def test_blueprint(self, model):
        model_copy = copy.deepcopy(model)

        self.memory_stress_test(model_copy)
        self.test_model_evaluation(model_copy)
        self.test_checkpoint_save_and_load(model_copy)
        self.estimate_training_time(model_copy)

        self.blueprint_completed_testing = True
        print("Congratulations, the blueprint test is complete!")

    def training_from_scratch(self, model):
        if not self.blueprint_completed_testing:
            print("The blueprint is not tested, of course you can train directly, but without testing why you design the blueprint..")

        n_dataloader = self.n_dataloader
        model_forward = self.model_forward
        log = self.log
        blueprint = self.blueprint
        model_eval = self.model_eval
        save_checkpoint = self.save_checkpoint
        eval_interval = blueprint.evaluation.interval_by_step
        checkpoint_interval = blueprint.checkpoint.interval_by_step

        optimizer = self.optimizer_constructor(model)
        lr_scheduler = self.lr_scheduler_constructor(optimizer)

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
                log(metrics=metrics, step=completed_steps)

                if completed_steps % eval_interval == 0:
                    eval_metrics, spend_time = model_eval(model)
                    log(metrics=eval_metrics, commit=False)

                if completed_steps % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, lr_scheduler)
