# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import time
import json
import logging
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy
from itertools import chain
from collections import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler, T_co
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from multiprocessing import Process, Queue
from unitorch import set_seed
from unitorch.cli import (
    register_task,
    registered_model,
    registered_optim,
    registered_dataset,
    registered_loss,
    registered_score,
    registered_scheduler,
    registered_writer,
    init_registered_module,
    init_registered_process,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models import BaseInputs, BaseOutputs, BaseTargets, LossOutputs

GENERATE_FINISHED = "done"
POSTPROCESS_FINISHED = None


class IOProcess(Process):
    """Write detokenized output to file in order."""

    def __init__(self, msg_queue, fout):
        """Async output writer.
        Args:
            msg_queue : Multiprocess message Queue
            fout : output file pointer.
        """
        super().__init__()
        self.msg_queue = msg_queue
        self.fout = fout
        self.waiting_for = 0
        self.string_buf = {}

    def process_string(self, string):
        self.fout.write(string)
        self.fout.flush()

    def process_buffer(self):
        while self.waiting_for in self.string_buf:
            self.process_string(self.string_buf[self.waiting_for])
            del self.string_buf[self.waiting_for]
            self.waiting_for += 1

    def run(self):
        while True:
            ind, string = self.msg_queue.get()
            if string == GENERATE_FINISHED:
                break
            elif ind != self.waiting_for:
                self.string_buf[ind] = string
            else:
                self.process_string(string)
                self.waiting_for += 1
                self.process_buffer()
        self.process_buffer()
        assert not self.string_buf, "IO Buffer not empty"
        self.msg_queue.close()
        self.msg_queue.join_thread()


class PostProcess(Process):
    """Parallel detokenization"""

    def __init__(
        self,
        post_process_fn,
        writer_fn,
        data_queue,
        msg_queue,
    ):
        """Async Postprocess.
        Args:
            data_queue : Multiprocess data Queue
            msg_queue :  Multiprocess message queue
        """
        super().__init__()
        self.data_queue = data_queue
        self.msg_queue = msg_queue
        self.post_process_fn = post_process_fn
        self.writer_fn = writer_fn

    def run(self):
        while True:
            ind, outputs = self.data_queue.get()
            if outputs == GENERATE_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED))
                break
            elif outputs == POSTPROCESS_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED))
                break
            else:
                string = self.writer_fn(
                    outputs=self.post_process_fn(outputs),
                )
                self.msg_queue.put((ind, string))

        self.data_queue.close()
        self.data_queue.join_thread()
        self.msg_queue.close()
        self.msg_queue.join_thread()


class DistributedSkipSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        skip_step: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.skip_step = skip_step

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[self.skip_step :])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step


class RandomSkipSampler(RandomSampler):
    def __init__(
        self,
        data_source,
        replacement=False,
        num_samples=None,
        skip_step=0,
    ):
        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
        )
        self.skip_step = skip_step

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n,
                    size=(self.num_samples,),
                    dtype=torch.int64,
                ).tolist()[self.skip_step :]
            )
        return iter(torch.randperm(n).tolist()[self.skip_step :])

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step


class SequentialSkipSampler(SequentialSampler):
    def __init__(
        self,
        data_source,
        skip_step=0,
    ):
        super().__init__(
            data_source=data_source,
        )
        self.skip_step = skip_step

    def __iter__(self):
        return iter(range(len(self.data_source))[self.skip_step :])

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step


def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return -1


def collate_fn(bufs):
    multi_inputs, multi_targets = list(zip(*bufs))
    if isinstance(multi_inputs[0], BaseInputs):
        inputs = type(multi_inputs[0]).from_list(*multi_inputs)
    else:
        multi_inputs = [type(_inputs[0]).from_list(*_inputs) for _inputs in list(zip(*multi_inputs))]
        inputs = BaseInputs()
        for _inputs in multi_inputs:
            inputs.update(_inputs)

    if isinstance(multi_targets[0], BaseTargets):
        targets = type(multi_targets[0]).from_list(*multi_targets)
    else:
        multi_targets = [type(_targets[0]).from_list(*_targets) for _targets in list(zip(*multi_targets))]
        targets = BaseTargets()
        for _targets in multi_targets:
            targets.update(_targets)

    return inputs, targets


class DatasetInfo(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        row = self.dataset[idx]
        ret = {}
        for k, v in row.items():
            if isinstance(v, Image.Image):
                v = np.array(v).tolist()
            if not isinstance(v, str):
                v = json.dumps(v)
            ret[k] = v
        return ret

    def __len__(self):
        return len(self.dataset)


@register_task("core/task/supervised_task")
class SupervisedTask(object):
    def __init__(
        self,
        __config__=None,
        model=None,
        datasets=None,
        seed=1123,
        local_rank=-1,
    ):
        set_seed(seed)
        self.n_gpu = 1 if torch.cuda.is_available() else 0
        if dist.is_initialized():
            self.n_gpu = dist.get_world_size()

        self.config = __config__
        self.model = model
        self.datasets = datasets
        self.local_rank = local_rank

        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.best_score = float("-inf")

    @classmethod
    @add_default_section_for_init("core/task/supervised_task")
    def from_core_configure(cls, config, **kwargs):
        try:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        except:
            logging.info("PyTorch is not in distributed mode")

        config.set_default_section("core/task/supervised_task")

        model = config.getoption("model", None)
        dataset = config.getoption("dataset", None)

        if model is not None:
            model = init_registered_module(model, config, registered_model)

        if dataset is not None:
            dataset = init_registered_module(dataset, config, registered_dataset)

        local_rank = config.getdefault(
            "core/cli",
            "local_rank",
            get_local_rank(),
        )

        return dict(
            __config__=config,
            model=model,
            datasets=dataset,
            local_rank=local_rank,
        )

    def monitor(self, outputs, targets, monitor_fns):
        if monitor_fns is None:
            return

        for monitor_fn in monitor_fns:
            score = monitor_fn(outputs=outputs, targets=targets)
            info = str(type(monitor_fn).__name__)
            logging.info(f"{info} is {score}")
        return

    @torch.no_grad()
    def score(self, iter_dev, score_fn):
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.eval()
        outputs, targets = [], []
        for _, (_inputs, _targets) in enumerate(iter_dev):
            if torch.cuda.is_available():
                _inputs = _inputs.cuda()
            _outputs = base_model(**_inputs.to_dict())
            outputs.append(_outputs.cpu())
            targets.append(_targets.cpu())

        if isinstance(outputs[0], LossOutputs):
            outputs = LossOutputs(
                loss=torch.tensor([output.loss for output in outputs]).to(device=outputs[0].loss.device)
            )
            if dist.is_initialized():
                outputs = outputs.cuda().sync().cpu()
            else:
                outputs = outputs.cpu()
        else:
            outputs = type(outputs[0]).from_list(*outputs, op="concat")
            targets = type(targets[0]).from_list(*targets, op="concat")

            if dist.is_initialized():
                outputs = outputs.cuda().sync().cpu()
                targets = targets.cuda().sync().cpu()
            else:
                outputs = outputs.cpu()
                targets = targets.cpu()

        new_score = score_fn(outputs=outputs, targets=targets)
        base_model.train()
        return BaseOutputs(
            score=new_score,
            outputs=outputs,
            targets=targets,
        )

    @torch.no_grad()
    def save_checkpoint(
        self,
        ckpt_dir,
        iter_dev,
        score_fn,
        monitor_fns,
        optim,
        scheduler,
        **kwargs,
    ):
        results = self.score(iter_dev, score_fn)
        if self.local_rank in [-1, 0]:
            self.monitor(
                outputs=results.outputs,
                targets=results.targets,
                monitor_fns=monitor_fns,
            )
            new_score = results.score
            if new_score > self.best_score:
                self.best_score = new_score
                base_model = self.model.module if hasattr(self.model, "module") else self.model
                base_model.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                )
                if optim:
                    optim.save_checkpoint(
                        ckpt_dir=ckpt_dir,
                    )

                if scheduler:
                    scheduler.save_checkpoint(
                        ckpt_dir=ckpt_dir,
                    )

            info_path = kwargs.pop("info_path", None)
            if info_path:
                base_model = self.model.module if hasattr(self.model, "module") else self.model
                base_model.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    weight_name="pytorch_model_latest.bin",
                )
                if optim:
                    optim.save_checkpoint(
                        ckpt_dir=ckpt_dir,
                        weight_name="pytorch_optim_latest.bin",
                    )
                if scheduler:
                    scheduler.save_checkpoint(
                        ckpt_dir=ckpt_dir,
                        weight_name="pytorch_scheduler_latest.bin",
                    )

                self.save_infos(info_path, best_score=self.best_score, **kwargs)

    def save_infos(self, info_path, **kwargs):
        json.dump(kwargs, open(info_path, "w"))

    @add_default_section_for_function("core/task/supervised_task")
    def train(
        self,
        optim=None,
        loss_fn=None,
        score_fn=None,
        monitor_fns=None,
        scheduler=None,
        from_ckpt_dir="./from_ckpt",
        to_ckpt_dir="./to_ckpt",
        train_batch_size=128,
        dev_batch_size=128,
        pin_memory=True,
        num_workers=4,
        save_optimizer=True,
        save_scheduler=True,
        log_freq=100,
        ckpt_freq=10000,
        grad_acc_step=1,
        max_grad_norm=1.0,
        num_training_samples=1000000000,
        epochs=5,
    ):
        if not os.path.exists(to_ckpt_dir) and self.local_rank in [-1, 0]:
            os.makedirs(to_ckpt_dir, exist_ok=True)

        if loss_fn is not None:
            loss_fn = init_registered_module(loss_fn, self.config, registered_loss)

        if score_fn is not None:
            score_fn = init_registered_module(score_fn, self.config, registered_score)

        if monitor_fns is not None:
            monitor_fns = [
                init_registered_module(monitor_fn, self.config, registered_score)
                for monitor_fn in monitor_fns
                if monitor_fn in registered_score
            ]

        if optim is not None and self.model is not None:
            optim = init_registered_module(
                optim,
                self.config,
                registered_optim,
                params=self.model.parameters(),
            )

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)
            optim.from_checkpoint(from_ckpt_dir)

        if os.path.exists(to_ckpt_dir):
            self.model.from_checkpoint(
                to_ckpt_dir,
                weight_name="pytorch_model_latest.bin",
            )
            optim.from_checkpoint(
                to_ckpt_dir,
                weight_name="pytorch_optim_latest.bin",
            )

        info_path = os.path.join(to_ckpt_dir, "info.json")
        if os.path.exists(info_path):
            info = json.load(open(os.path.join(to_ckpt_dir, "info.json")))
        else:
            info = dict()

        global_epoch = info.get("global_epoch", 0)
        global_step = info.get("global_step", 0)
        self.best_score = info.get("best_score", self.best_score)

        logging.info(f"the best score is {self.best_score}")
        """
        _tensor = torch.tensor([global_epoch, global_step], dtype=torch.int32)
        if torch.cuda.is_available():
            _tensor = _tensor.cuda()

        if dist.is_initialized():
            dist.broadcast(_tensor, 0)

        global_epoch, global_step = int(_tensor[0]), int(_tensor[1])
        """

        global_rank = -1
        if self.n_gpu > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            global_rank = dist.get_rank()

        train_sampler = DistributedSkipSampler if self.n_gpu > 1 else RandomSkipSampler
        dev_sampler = DistributedSampler if self.n_gpu > 1 else SequentialSampler

        dataset_train = self.datasets.get("train")
        iter_train = DataLoader(
            dataset_train,
            sampler=train_sampler(dataset_train) if not isinstance(dataset_train, Iterable) else None,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        dataset_dev = self.datasets.get("dev")
        iter_dev = DataLoader(
            dataset_dev,
            sampler=dev_sampler(dataset_dev) if not isinstance(dataset_dev, Iterable) else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        if scheduler is not None:
            if not isinstance(dataset_train, Iterable):
                num_training_steps = int(
                    epochs * len(dataset_train) // train_batch_size // max(1, self.n_gpu) // grad_acc_step
                )
            else:
                num_training_steps = int(
                    epochs * num_training_samples // train_batch_size // max(1, self.n_gpu) // grad_acc_step
                )

            scheduler = init_registered_module(
                scheduler,
                self.config,
                registered_scheduler,
                optimizer=optim,
                num_training_steps=num_training_steps,
            )
        """
        if scheduler and os.path.exists(from_ckpt_dir):
            scheduler.from_checkpoint(from_ckpt_dir)
        """

        if scheduler and os.path.exists(to_ckpt_dir):
            scheduler.from_checkpoint(
                to_ckpt_dir,
                weight_name="pytorch_scheduler_latest.bin",
            )

        scaler = GradScaler()

        log_loss = 0
        dev_epoch = 0
        for e in range(0, epochs):
            torch.cuda.empty_cache()
            if e < global_epoch:
                continue

            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(e)

            if hasattr(dataset_train, "set_skip_step"):
                dataset_train.set_skip_step(global_step * train_batch_size)

            if hasattr(iter_train.sampler, "set_epoch"):
                iter_train.sampler.set_epoch(e)

            if hasattr(iter_train.sampler, "set_skip_step"):
                iter_train.sampler.set_skip_step(global_step * train_batch_size)

            self.model.train()
            is_update_step = False
            for step, (inputs, targets) in enumerate(iter_train):
                step = step + global_step
                is_update_step = False
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                with autocast():
                    outputs = self.model(**inputs.to_dict())
                    if isinstance(outputs, LossOutputs):
                        loss = outputs.loss / grad_acc_step
                    else:
                        loss = loss_fn(outputs=outputs, targets=targets) / grad_acc_step

                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                scaler.scale(loss).backward()

                log_loss += loss.data * grad_acc_step
                if (step + 1) % grad_acc_step == 0:
                    is_update_step = True
                    scaler.step(optim)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optim.zero_grad()

                if (step + 1) % log_freq == 0 and global_rank in [-1, 0]:
                    logging.info(f"epoch {e} step {step}: loss -- { log_loss / log_freq }")
                    log_loss = 0

                if (step + 1) % ckpt_freq == 0:
                    if hasattr(dataset_dev, "set_epoch"):
                        dataset_dev.set_epoch(dev_epoch)

                    if hasattr(iter_dev.sampler, "set_epoch"):
                        iter_dev.sampler.set_epoch(dev_epoch)

                    dev_epoch += 1
                    self.save_checkpoint(
                        to_ckpt_dir,
                        iter_dev,
                        score_fn,
                        monitor_fns,
                        optim=optim if save_optimizer else None,
                        scheduler=scheduler if save_scheduler else None,
                        info_path=info_path,
                        global_epoch=e,
                        global_step=step + 1,
                    )

            if not is_update_step:
                scaler.step(optim)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad()

            log_loss = 0

            if hasattr(dataset_dev, "set_epoch"):
                dataset_dev.set_epoch(dev_epoch)

            if hasattr(iter_dev.sampler, "set_epoch"):
                iter_dev.sampler.set_epoch(dev_epoch)

            dev_epoch += 1

            global_step = 0
            self.save_checkpoint(
                to_ckpt_dir,
                iter_dev,
                score_fn,
                monitor_fns,
                optim=optim if save_optimizer else None,
                scheduler=scheduler if save_scheduler else None,
                info_path=info_path,
                global_epoch=e + 1,
                global_step=0,
            )

    @torch.no_grad()
    @add_default_section_for_function("core/task/supervised_task")
    def infer(
        self,
        output_header: Optional[List] = None,
        test_batch_size: Optional[int] = 128,
        pin_memory: bool = True,
        num_workers: Optional[int] = 4,
        max_size: Optional[int] = 10000,
        from_ckpt_dir: str = "./from_ckpt",
        output_path: str = "./cache/predict.txt",
        postprocess_workers: Optional[int] = 2,
        nrows_per_sample: Optional[int] = None,
        post_process_fn: str = None,
        writer_fn: str = None,
    ):
        assert self.n_gpu <= 1
        assert writer_fn is not None

        if post_process_fn is not None:
            post_process_fn = init_registered_process(post_process_fn, self.config)

        if writer_fn is not None:
            writer_fn = init_registered_module(
                writer_fn,
                self.config,
                registered_writer,
            )

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)

        if nrows_per_sample is None:
            sampler = SequentialSampler
        else:
            sampler = SequentialSkipSampler

        dataset_test = self.datasets.get("test")
        iter_test = DataLoader(
            dataset_test,
            sampler=sampler(dataset_test) if not isinstance(dataset_test, Iterable) else None,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        skip_step = (
            0
            if nrows_per_sample is None or not os.path.exists(output_path)
            else sum(1 for _ in open(output_path)) // nrows_per_sample
        )
        if skip_step > 0:
            output_file = open(output_path, "a")
        else:
            output_file = open(output_path, "w")

        if skip_step > 0 and hasattr(dataset_test, "set_skip_step"):
            dataset_test.set_skip_step(skip_step)

        if skip_step > 0 and hasattr(iter_test.sampler, "set_skip_step"):
            iter_test.sampler.set_skip_step(skip_step)

        if hasattr(dataset_test, "dataset"):
            data_info = dataset_test.dataset
            data_info = DatasetInfo(data_info)
            iter_data = DataLoader(
                deepcopy(data_info),
                sampler=sampler(data_info) if not isinstance(dataset_test, Iterable) else None,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=None,
            )
        else:
            iter_data = None

        if skip_step > 0 and hasattr(iter_data.sampler, "set_skip_step"):
            iter_data.sampler.set_skip_step(skip_step)

        self.model.eval()
        start = time.time()

        data_queue = Queue(maxsize=max_size)
        msg_queue = Queue(maxsize=max_size)
        post_process_list = []
        for _ in range(postprocess_workers):
            p = PostProcess(
                post_process_fn,
                writer_fn,
                data_queue,
                msg_queue,
            )
            post_process_list.append(p)
            p.start()

        io_process = IOProcess(msg_queue, output_file)
        io_process.start()

        if iter_data is None:
            for step, (inputs, _) in enumerate(iter_test):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.to_dict())
                outputs = outputs.cpu()
                data_queue.put((step, outputs))
        else:
            for step, ((inputs, _), _infos) in enumerate(zip(iter_test, iter_data)):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.to_dict())
                outputs = outputs.cpu()
                if output_header is not None:
                    _infos = {k: _infos[k] for k in output_header if k in _infos}
                    outputs.update(_infos)
                data_queue.put((step, outputs))

        data_queue.put((-1, GENERATE_FINISHED))
        for p in post_process_list:
            p.join()

        msg_queue.put((-1, GENERATE_FINISHED))
        io_process.join()

        end = time.time()
        ms = (end - start) * 1000
        logging.info(
            "{:.2f} ms, {:.1f} sample/s".format(
                ms,
                ((len(dataset_test) - skip_step) / ms * 1000),
            )
        )
