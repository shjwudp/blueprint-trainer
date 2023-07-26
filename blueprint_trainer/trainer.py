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

import time
import yaml
import copy
import os
import random
import contextlib

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
import tabulate
from omegaconf import OmegaConf


class Trainer:
    def __init__(
        self,
        blueprint_text=None,
        blueprint_filepath=None,
    ) -> None:
        if blueprint_text:
            blueprint_conf = OmegaConf.create(blueprint_text)
        elif blueprint_filepath:
            blueprint_conf = OmegaConf.load(blueprint_filepath)

        self.blueprint = blueprint_conf.blueprint
        self.blueprint_detail = OmegaConf.create({
            "total_training_steps": None,
            "time_cost_of_each_stage": None,
            "gpu_util_at_each_stage": None,
        })

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
        self.blueprint_detail.total_training_steps = sum(len(dl)//step for dl, step in zip(self.n_dataloader, self.n_step_of_gradient_accumulation))

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

    def print_blueprint(self, model):
        import plotext as plt
        
        total_training_steps = self.blueprint_detail.total_training_steps

        optimizer = self.optimizer_constructor(model)
        lr_scheduler = self.lr_scheduler_constructor(optimizer)
        learning_rates = []
        for i in range(total_training_steps):
            lr = lr_scheduler.get_lr()
            assert len(lr) == 1, "Currently only lr scheduler with a length of 1 is supported, if this is not your usage scenario, please report bugs at https://github.com/shjwudp/blueprint-trainer/issues"
            learning_rates.append(lr[0])

            lr_scheduler.step()
            
        plt.scatter(learning_rates)
        plt.title("learning rate")
        plt.show()
        print(Omegaconf.to_yaml(self.blueprint))
        print(Omegaconf.to_yaml(self.blueprint_detail))

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
        time_mark = time.time()
        large_nbatch_quantity = 0
        large_nbatch = []
        nbatch = []
        for batch in dataloader:
            nbatch.append(batch)
            if len(nbatch) == step_of_gradient_accumulation:
                q = sum([sum(x.numel() for x in batch.values()) for batch in nbatch])
                if q > large_nbatch_quantity:
                    large_nbatch = nbatch
                    large_nbatch_quantity = q
                nbatch = []

        return large_nbatch
    
    def _dp_no_sync(self, model):
        if self.dp_handler:
            return model.no_sync
        return contextlib.nullcontext
    
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
                with accumulator.accumulate(self._dp_no_sync(model)):
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

    def memory_stress_test(self, model, dataset):
        print("Start Memory Stress Test..")
        function_start_time = time.time()
        n_dataloader = self.n_dataloader

        # Memory Stress Test
        micro_batch_solutions = []
        for i, plan in enumerate(self.blueprint.batch_size_plan):
            batch_size = plan.batch_size

            # Find out among existing solutions
            step_of_gradient_accumulation = 0
            for mb in micro_batch_solutions:
                if batch_size % mb == 0:
                    step_of_gradient_accumulation = batch_size//mb

            # Find out in experiments
            if step_of_gradient_accumulation == 0:
                dl = self._get_dataloader(dataset, batch_size)
                step_of_gradient_accumulation = self.\
                    _tuning_step_of_gradient_accumulation(model, dl)
                
            if i >= len(self.n_step_of_gradient_accumulation):
                break
                
            self.n_step_of_gradient_accumulation[i] = step_of_gradient_accumulation
            micro_batch = batch_size//step_of_gradient_accumulation
            self.n_dataloader[i] = self._get_dataloader(
                self.n_dataloader[i].dataset,
                micro_batch,
            )
            micro_batch_solutions.append(micro_batch)
            
        print(f"Memory Stress Test done. It takes {seconds_to_human_friendly_time_str(time.time()-function_start_time)}.")
        table_str = tabulate.tabulate(
            list(zip(
                list(range(1, len(self.n_dataloader)+1)),
                [plan.batch_size for plan in self.blueprint.batch_size_plan][:len(n_dataloader)],
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
        metrics, spend_time = model_eval(model)
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

    def estimate_training_time(self, model, dataset):
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
        N_SAMPLES = 1
        time_cost_of_each_stage = []
        gpu_util_at_each_stage = []
        for i, step in enumerate(self.n_step_of_gradient_accumulation):
            print(f"Test #{i+1} of {len(n_dataloader)} Dataloader..")
            batch_size = self.blueprint.batch_size_plan[i].batch_size
            micro_batch = batch_size//step
            dl = self._get_dataloader(dataset, micro_batch)

            accumulator = blueprint_utils.GradientAccumulator(step)
            sample_count = 0
            sample_time = 0
            time_mark = time.time()
            for batch in dl:
                with accumulator.accumulate(self._dp_no_sync(model)):
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
                    if sample_count > N_SAMPLES:
                        break

            batch_count = len(self.n_dataloader[i])
            estimated_training_time = batch_count * (sample_time / (sample_count*step))
            time_cost_of_each_stage.append(estimated_training_time)

            gpu_util_at_each_stage.append(gpu_util.aggregate())
            gpu_util.clear()

        self.blueprint_detail.time_cost_of_each_stage = time_cost_of_each_stage
        self.blueprint_detail.gpu_util_at_each_stage = gpu_util_at_each_stage

        print(f"Estimate Training Time Done. It takes {seconds_to_human_friendly_time_str(time.time()-function_start_time)}. The training is expected to take {seconds_to_human_friendly_time_str(sum(time_cost_of_each_stage))}.")
        table_str = tabulate.tabulate(
            list(zip(
                list(range(1, len(n_dataloader)+1)),
                [plan.batch_size for plan in self.blueprint.batch_size_plan][:len(n_dataloader)],
                [seconds_to_human_friendly_time_str(secs) for secs in time_cost_of_each_stage],
                gpu_util_at_each_stage,
            )),
            headers=["#", "Batch Size", "Estimated Training Time", "GPU Util"],
        )
        print(table_str)

    def _sample_dataset_for_testing(self):
        max_batch_size = 0
        for plan in self.blueprint.batch_size_plan:
            max_batch_size = max(max_batch_size, plan.batch_size)
            
        num_samples = max_batch_size * 5
        samples_index = random.sample(list(range(len(self.train_dataset))), num_samples)
        sampled_dataset = Subset(self.train_dataset, samples_index)
        
        return sampled_dataset

    def test_blueprint(self, model):
        start_timestamp = time.time()
        model_copy = copy.deepcopy(model)

        self.test_checkpoint_save_and_load(model_copy)
        self.test_model_evaluation(model_copy)
        
        sampled_dataset = self._sample_dataset_for_testing()
        self.memory_stress_test(model_copy, sampled_dataset)
        self.estimate_training_time(model_copy, sampled_dataset)

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
        ckpt_dir = self.blueprint.checkpoint.path

        optimizer = self.optimizer_constructor(model)
        lr_scheduler = self.lr_scheduler_constructor(optimizer)

        # model training
        completed_steps = 0
        optimizer.zero_grad()
        time_stone = time.time()
        for dl, n_gas in zip(n_dataloader, self.n_step_of_gradient_accumulation):
            accumulator = blueprint_utils.GradientAccumulator(n_gas)
            for batch in dl:
                with accumulator.accumulate(self._dp_no_sync(model)):
                    loss, metrics = model_forward(model, batch)
                    loss.backward()

                if accumulator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    completed_steps += 1

                    # logging
                    spend_time = time.time() - time_stone
                    time_stone = time.time()
                    metrics.update({"loss": loss, "spend_time": spend_time, "lr": lr_scheduler.get_lr()})
                    log(metrics=metrics, step=completed_steps)

                    if completed_steps % eval_interval == 0:
                        eval_metrics, spend_time = model_eval(model)
                        log(metrics=eval_metrics, commit=False)

                    if completed_steps % checkpoint_interval == 0:
                        save_checkpoint(
                            ckpt_dir=ckpt_dir,
                            step=completed_steps,
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                        )
