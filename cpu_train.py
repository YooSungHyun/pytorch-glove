import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

import json
import numpy as np
import torch
import wandb
from arguments.training_args import TrainingArguments
from networks.models import GloVeModel
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from torch.utils.data import RandomSampler, SequentialSampler, random_split
from trainer.cpu import Trainer
from utils.comfy import (
    apply_to_collection,
    dataclass_to_namespace,
    seed_everything,
    tensor_dict_to_device,
    web_log_every_n,
)
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import LengthGroupedSampler
from utils.data.nlp_dataset import CBOWDataset
import random
from collections import defaultdict, deque

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


# TODO(User): override training_step and eval_loop for your style
class CPUTrainer(Trainer):
    def __init__(
        self,
        criterion,
        eval_metric=None,
        precision="fp32",
        cmd_logger=None,
        web_logger=None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        chk_addr_dict: dict = None,
        non_blocking: bool = True,
        log_every_n: int = 1,
    ):
        super().__init__(
            criterion,
            eval_metric,
            precision,
            cmd_logger,
            web_logger,
            max_epochs,
            max_steps,
            grad_accum_steps,
            limit_train_batches,
            limit_val_batches,
            validation_frequency,
            checkpoint_dir,
            checkpoint_frequency,
            chk_addr_dict,
            non_blocking,
            log_every_n,
        )

    def training_step(self, model, batch, batch_idx) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: model to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        labels = batch.pop("prob")
        focal_embed, context_embed, focal_bias, context_bias = model(**batch)
        loss = self.criterion(focal_embed, context_embed, focal_bias, context_bias, labels)

        def on_before_backward(loss):
            pass

        on_before_backward(loss)
        loss.backward()

        def on_after_backward():
            pass

        on_after_backward()

        outputs = {"loss": loss}
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        web_log_every_n(
            self.web_logger,
            {
                "train/loss": self._current_train_return["loss"],
                "train/step": self.step,
                "train/global_step": self.global_step,
                "train/epoch": self.current_epoch,
            },
            self.step,
            self.log_every_n,
        )
        return loss


def main(hparams: TrainingArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    web_logger = wandb.init(config=hparams)
    seed_everything(hparams.seed)

    train_dataset = np.loadtxt("./raw_data/train_comat.npy", dtype=np.float32)
    eval_dataset = np.loadtxt("./raw_data/eval_comat.npy", dtype=np.float32)
    logger.info(train_dataset)

    with open("raw_data/glove_vocab.json", "r") as st_json:
        tokenizer = json.load(st_json)

    train_dataset = CBOWDataset(train_dataset, tokenizer)
    eval_dataset = CBOWDataset(eval_dataset, tokenizer)

    # Instantiate objects
    model = GloVeModel(vocab_size=len(tokenizer.keys()), embedding_size=512)
    web_logger.watch(model, log_freq=hparams.log_every_n)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )

    generator = None
    custom_train_sampler = RandomSampler(train_dataset, generator=generator)
    custom_eval_sampler = SequentialSampler(eval_dataset)

    # If 1 device for training, sampler suffle True and dataloader shuffle True is same meaning
    train_dataloader = CustomDataLoader(
        dataset=train_dataset,
        batch_size=hparams.per_device_train_batch_size,
        sampler=custom_train_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
    )

    eval_dataloader = CustomDataLoader(
        dataset=eval_dataset,
        batch_size=hparams.per_device_eval_batch_size,
        sampler=custom_eval_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
    )

    # dataloader already calculate total_data / batch_size
    # accumulation is always floor
    train_steps_per_epoch = math.floor(len(train_dataloader) / (hparams.accumulate_grad_batches))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=train_steps_per_epoch,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": None}
    assert id(scheduler) == id(lr_scheduler["scheduler"])
    criterion = model._loss
    trainable_loss = None

    # I think some addr is same into trainer init&fit respectfully
    chk_addr_dict = {
        "train_dataloader": id(train_dataloader),
        "eval_dataloader": id(eval_dataloader),
        "model": id(model),
        "optimizer": id(optimizer),
        "criterion": id(criterion),
        "scheduler_cfg": id(lr_scheduler),
        "scheduler_cfg[scheduler]": id(lr_scheduler["scheduler"]),
        "trainable_loss": id(trainable_loss),
    }

    log_str = f"""\n##########################################
    train_dataloader addr: {chk_addr_dict["train_dataloader"]}
    eval_dataloader addr: {chk_addr_dict["eval_dataloader"]}
    model addr: {chk_addr_dict["model"]}
    optimizer addr: {chk_addr_dict["optimizer"]}
    criterion addr: {chk_addr_dict["criterion"]}
    scheduler_cfg addr: {chk_addr_dict["scheduler_cfg"]}
    scheduler addr: {chk_addr_dict["scheduler_cfg[scheduler]"]}
    ##########################################
    """
    logger.debug(log_str)
    # TODO(User): input your eval_metric
    eval_metric = None
    trainer = CPUTrainer(
        criterion=criterion,
        eval_metric=eval_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        web_logger=web_logger,
        max_epochs=hparams.max_epochs,
        grad_accum_steps=hparams.accumulate_grad_batches,
        chk_addr_dict=chk_addr_dict,
        checkpoint_dir=hparams.output_dir,
        log_every_n=hparams.log_every_n,
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
        trainable_loss=trainable_loss,
    )

    web_logger.finish(exit_code=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
