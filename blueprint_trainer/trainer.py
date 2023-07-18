from blueprint_trainer.utils import (
    are_the_models_the_same,
    are_the_optimizers_the_same,
    are_the_lr_scheduler_the_same,
    seconds_to_human_friendly_time_str,
)
import blueprint_trainer.utils as blueprint_utils
from blueprint_trainer.algorithm import lambda_is_ok_upper_bound_int
from blueprint_trainer.system import (
    pynvml,
    GPUUtilization,
)

from collections import namedtuple
import time
import yaml
import copy
import os

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
import tabulate


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

        for plan in blueprint.batch_size_plan:
            bs, train_step = plan.batch_size, plan.training_nsteps
            if train_step == -1 or bs*train_step > len(train_dataset):
                dataloader = self._get_dataloader(train_dataset, bs)
                n_dataloader.append(dataloader)
                break

            stage_dataset = Subset(train_dataset, range(bs*train_step))
            train_dataset = Subset(train_dataset, range(bs*train_step, len(train_dataset)))

            dataloader = self._get_dataloader(stage_dataset, bs)
            n_dataloader.append(dataloader)
        self.n_dataloader = n_dataloader
        self.n_step_of_gradient_accumulation = [1]*len(self.n_dataloader)

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

    def _get_dataloader(self, dataset, batch_size):
        dl_kwargs = copy.copy(self.dataloader_kwargs)
        dp = self.dp_handler
        if dp:
            # TODO: Supports data parallelism with uneven batch size 
            # distribution, and variable-length data sets have such 
            # usage scenarios
            assert batch_size % dp.world_size == 0
            assert "sampler" not in dl_kwargs
            dl_kwargs["sampler"] = DistributedSampler(
                dataset, dp.world_size, dp.rank, shuffle=False
            )
            batch_size = batch_size//dp.world_size

        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            **dl_kwargs,
        )

        return dataloader
    
    def _get_large_nbatch_from_dataloader(
        self,
        dataloader,
        step_of_gradient_accumulation,
    ):
        large_nbatch_quantity = 0
        large_nbatch = []
        nbatch = []
        for batch in dataloader:
            nbatch.append(batch)
            if len(nbatch) == step_of_gradient_accumulation:
                q = [sum(x.numel() for x in batch.values()) for batch in nbatch]
                if q > large_nbatch_quantity:
                    large_nbatch = nbatch
                nbatch = []

        return large_nbatch
    
    def _test_step_of_gradient_accumulation(
        self,
        model,
        dataloader,
        step_of_gradient_accumulation,
    ):
        optimizer = self.optimizer_constructor(model)
        model_forward = self.model_forward
        lr_scheduler = self.lr_scheduler_constructor(optimizer)
        accumulator = blueprint_utils.GradientAccumulator(step_of_gradient_accumulation)

        large_nbatch = self._get_large_nbatch_from_dataloader(
            dataloader,
            step_of_gradient_accumulation,
        )

        for batch in large_nbatch:
            try:
                with accumulator.accumulate(model.no_sync):
                    loss, _ = model_forward(model, batch)
                    loss.backward()

                if accumulator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            except torch.cuda.OutOfMemoryError:
                return False
            except:
                raise

        return True

    def _tuning_step_of_gradient_accumulation(
        self,
        model,
        dataloader,
    ):
        batch_size = dataloader.batch_size
        micro_batches = list(filter(lambda x: batch_size % x == 0, range(1, batch_size + 1)))

        def is_gradient_accumulation_step_ok(index):
            micro_batch = micro_batches[index]
            step_of_gradient_accumulation = batch_size//micro_batch

            dl = self._get_dataloader(dataloader.dataset, micro_batch)

            return self._test_step_of_gradient_accumulation(
                model,
                dl,
                step_of_gradient_accumulation,
            )

        index = lambda_is_ok_upper_bound_int(
            is_gradient_accumulation_step_ok,
            0, len(micro_batches)-1,
        ) - 1

        if index < 0:
            raise torch.cuda.OutOfMemoryError("batch_size = 1 memory limit exceeded!!")

        step_of_gradient_accumulation = batch_size//micro_batches[index]

        return step_of_gradient_accumulation

    def memory_stress_test(self, model):
        print("Start Memory Stress Test..")
        function_start_time = time.time()
        n_dataloader = self.n_dataloader

        # Memory Stress Test
        suggest_micro_batch = n_dataloader[0].batch_size
        for i, dl in enumerate(n_dataloader):
            print(f"Test #{i+1} of {len(n_dataloader)} Dataloader..")

            step_of_gradient_accumulation = 0
            if dl.batch_size % suggest_micro_batch == 0:
                step_of_gradient_accumulation = dl.batch_size//suggest_micro_batch
                test_dl = self._get_dataloader(dl.dataset, suggest_micro_batch)
                if not self._test_step_of_gradient_accumulation(model, test_dl, step_of_gradient_accumulation):
                    step_of_gradient_accumulation = 0

            if not step_of_gradient_accumulation:
                step_of_gradient_accumulation = self.\
                    _tuning_step_of_gradient_accumulation(model, dl)

            self.n_step_of_gradient_accumulation[i] = step_of_gradient_accumulation
            micro_batch = dl.batch_size//step_of_gradient_accumulation
            self.n_dataloader[i] = self._get_dataloader(
                dl.dataset,
                micro_batch,
            )
            suggest_micro_batch = micro_batch
        print(f"Memory Stress Test done. It takes {seconds_to_human_friendly_time_str(time.time()-function_start_time)}.")
        table_str = tabulate.tabulate(
            list(zip(
                list(range(1, len(n_dataloader)+1)),
                [dl.batch_size for dl in n_dataloader],
                self.n_step_of_gradient_accumulation,
            )),
            headers=["#", "Batch Size", "Step of Gradient Accumulation"],
        )
        print(table_str)

    def test_model_evaluation(self, model):
        print("Start Model Evaluation Test..")
        function_start_time = time.time()
        model_eval = self.model_eval

        # Test Model Evaluation
        model_eval(model)
        print(f"Model Evaluation Test done. It takes {seconds_to_human_friendly_time_str(time.time()-function_start_time)}.")

    def test_checkpoint_save_and_load(self, model):
        print("Start Checkpoint Save & Load Test..")
        function_start_time = time.time()
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
        print(f"Checkpoint Save & Load Test done. It takes {seconds_to_human_friendly_time_str(time.time()-function_start_time)}.")

    def estimate_training_time(self, model):
        print("Start Estimate Training Time..")
        function_start_time = time.time()
        estimated_training_time = 0
        optimizer = self.optimizer_constructor(model)
        n_dataloader = self.n_dataloader
        model_forward = self.model_forward
        lr_scheduler = self.lr_scheduler_constructor(optimizer)

        # Init gpu monitor
        pynvml.nvmlInit()
        gpu_util = GPUUtilization(os.getpid())

        # Estimate Training Time
        optimizer.zero_grad()
        N_SAMPLES = 3
        for dl, n_sga in zip(n_dataloader, self.n_step_of_gradient_accumulation):
            accumulator = blueprint_utils.GradientAccumulator(n_sga)
            batch_count = 0
            sample_count = 0
            sample_time = 0
            time_mark = time.time()
            for batch in dl:
                batch_count += 1

                if sample_count > N_SAMPLES:
                    continue

                with accumulator.accumulate(model.no_sync):
                    loss, _ = model_forward(model, batch)
                    loss.backward()
                    gpu_util.sample()

                if accumulator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    sample_time += time.time() - time_mark
                    time_mark = time.time()
                    sample_count += 1

            if sample_count:
                estimated_training_time += batch_count * (sample_time / (sample_count*self.n_step_of_gradient_accumulation))

        print(f"Estimate Training Time Done. It takes {seconds_to_human_friendly_time_str(time.time()-function_start_time)}. The training is expected to take {seconds_to_human_friendly_time_str(estimated_training_time)}.")
        print(f"When estimating the training time, the GPU util was sampled by the way, and the sampling result is: {gpu_util.aggregate()}")

    def test_blueprint(self, model):
        start_timestamp = time.time()
        model_copy = copy.deepcopy(model)

        self.test_checkpoint_save_and_load(model_copy)
        self.test_model_evaluation(model_copy)
        self.memory_stress_test(model_copy)
        self.estimate_training_time(model_copy)

        self.blueprint_completed_testing = True
        print(f"Congratulations, the blueprint test is complete! It takes {seconds_to_human_friendly_time_str(time.time()-start_timestamp)}.")

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
        for dl, n_gas in zip(n_dataloader, self.n_step_of_gradient_accumulation):
            accumulator = blueprint_utils.GradientAccumulator(n_gas)
            for batch in dl:
                with accumulator.accumulate(model.no_sync):
                    loss, metrics = model_forward(model, batch)
                    loss.backward()

                if accumulator.sync_gradients:
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
