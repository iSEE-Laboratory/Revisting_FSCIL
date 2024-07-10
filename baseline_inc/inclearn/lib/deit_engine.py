# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import inclearn.lib.utils as utils

from inclearn.lib.logger import LOGGER
logger = LOGGER.LOGGER


def train_one_epoch_baseline_notea(model: torch.nn.Module, criterion,
                                   data_loader: Iterable, optimizer: torch.optim.Optimizer,
                                   device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                                   model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                                   set_training_mode=True, ):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(data_loader, print_freq, header):
        samples = data['inputs']
        targets = data['targets']
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, feat=True)
            outputs = outputs[0]['logits']
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_baseline(model: torch.nn.Module,
                             data_loader: Iterable,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             loss_scaler,
                             max_norm: float = 0,
                             mixup_fn: Optional[Mixup] = None,
                             set_training_mode=True,
                             print_freq=10,
                             nb_classes=50,
                             teacher_model=None,
                             ):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    data_idx = 0
    for data in metric_logger.log_every(data_loader, print_freq, header):
        data_idx += 1
        loss = 0
        samples = data['inputs']
        targets = data['targets']
        samples = samples.to(device, non_blocking=True)
        origin_t = targets = targets.to(device, non_blocking=True)

        bs, c, h, w = samples.shape
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if teacher_model is not None:
            teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = teacher_model(samples)
                teacher_output_logits = outputs_teacher['logits']
                _teacher_targets = teacher_output_logits.argmax(1)
        with torch.cuda.amp.autocast():
            outputs = model(samples, feat=True)
            logits = outputs[0]['logits']

            # loss = criterion(logits, targets)  # global
            loss = torch.zeros([]).to(device)
            metric_logger.update(**{f'global': loss.item()})

            if teacher_model is not None:
                teacher_model.eval()
                with torch.no_grad():
                    outputs_teacher = teacher_model(samples)
                    teacher_output_logits = outputs_teacher['logits']
                    _teacher_targets = teacher_output_logits.argmax(1)
                # onehot = F.one_hot(_teacher_targets, nb_classes)
                # logits = logits - onehot * margin * temperature
                _tea_dis_loss = F.cross_entropy(logits, _teacher_targets)
                loss += _tea_dis_loss
                metric_logger.update(**{f'dis_local': _tea_dis_loss.item()}, )
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.warning("Loss is {}, stopping training".format(loss_value))
            logger.warning({k: meter.global_avg for k, meter in metric_logger.meters.items()})
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        # if model_ema is not None:
        #     model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # break # used in debugging
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{str(metric_logger)}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pure(data_loader, model, device, use_aug=False, nb_classes=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    print(f"use_aug: {use_aug}")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # torch.distributed.barrier()
    all_pred = torch.tensor([])
    all_targets = torch.tensor([])
    for data in metric_logger.log_every(data_loader, 10, header):
        images = data['inputs']
        targets = data['targets']
        images = images.to(device, non_blocking=True)
        target = targets.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        output = model(images)
        output = output['logits']
        if nb_classes is not None:
            output = output[:, :nb_classes]

        loss = criterion(output, target)
        all_pred = torch.cat((all_pred, output.cpu()))
        all_targets = torch.cat((all_targets, targets.cpu()))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # all_pred = numpy.asarray(all_pred)
    # all_targets = numpy.asarray(all_targets).astype(int)
    # all_acc = accuracy_per_task(all_pred, all_targets, init_task_size=init_task, task_size=task_size, topk=1)
    # logger.info(f'---' * 50)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
