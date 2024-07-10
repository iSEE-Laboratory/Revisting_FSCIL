import collections
import copy
import os
import pickle
import pprint

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F

from inclearn.lib import network, utils
from inclearn.models.base import IncrementalLearner
from inclearn.lib.deit_engine import evaluate_pure, train_one_epoch_baseline, train_one_epoch_baseline_notea
from timm.data import Mixup
from timm.data.mixup import mixup_target
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import argparse
import json
import time, datetime
from inclearn.lib.metrics import _accuracy_per_task as accuracy_per_task
from collections import OrderedDict
from inclearn.backbones.etf_resnet.etf_model import etf_model

EPSILON = 1e-8

from inclearn.lib.logger import LOGGER

logger = LOGGER.LOGGER


class VitBaselineFewshot(IncrementalLearner):
    """Training transformers in a continual manner.

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args: dict):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["opt"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]
        self._evaluation_type = args.get("eval_type", "icarl")

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args["validation"]
        self.args = args  # type:dict
        self.clip_grad = args.get("clip_grad", None)
        self._result_folder = None
        self.testing_interval = args.get('testing_interval', 1)
        self.fc_proxy_per_class = args.get("fc_proxy_per_class", 1)

        self._network = network.BasicNet(
            args["backbones"],
            classifier_kwargs=args.get("classifier_config",
                                       {"type": "fc",
                                        "use_bias": True,
                                        }),
            device=self._device, extract_no_act=True, classifier_no_act=False,
            all_args=args,
            ddp=True)

        self.smoothing = args.get("smoothing", 0)

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None

        self._epoch_metrics = collections.defaultdict(list)

        self.parameter_dict = self.args.get('parameter_dict', {})

        self.all_centroid = torch.tensor([])
        # pretrain
        self.pretrain_config = args.get('pretrain_config', {})
        self.pretrain_lr_config = self.pretrain_config.get("lr_config", {})
        self.pretrain_epochs = self.pretrain_lr_config.get('epochs', 150)

        self.pretrain_lr_config['scaled'] = False
        self.pretrain_lr_config_args = argparse.Namespace(**self.pretrain_lr_config)
        self.pretrain_optimizer = create_optimizer(self.pretrain_lr_config_args, self._network)
        self.pretrain_scheduler, _ = create_scheduler(self.pretrain_lr_config_args, self.pretrain_optimizer)

        # teacher
        self.teacher_state_dict_path = args.get('teacher_path', None)
        self.use_teacher_distillate = args.get('use_resnet_teacher', False)
        if self.use_teacher_distillate and 'convnet' in args['convnet']:
            cnn_classifier_kwargs = args.get("cnn_classifier_config", {})
            self.teacher_network = network.BasicNet(
                args["convnet"],
                convnet_kwargs=args.get("convnet_config", {}),
                classifier_kwargs=cnn_classifier_kwargs,
                postprocessor_kwargs=args.get("cnn_postprocessor_config", {}),
                device=self._device,
                return_features=True,
                extract_no_act=True,
                classifier_no_act=args.get("classifier_no_act", True),
                all_args=self.args
            )
        elif self.use_teacher_distillate and 'neural_collapse' in args['convnet']:
            self.teacher_network = etf_model(args.get('teacher_path')).cuda()
        else:
            self.teacher_network = None

        self.ddp = True
        self.local_mixup = args.get('local_mixup', False)

    def load_teacher_statedict(self, state_dict_path):
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_k = k
                if 'stage_' in k:
                    new_k = k.replace('stage_', 'layer')
                if 'convnet' in k:
                    new_k = new_k.replace('convnet', 'backbone')
                if 'downsample.conv1x1' in k:
                    new_k = new_k.replace('downsample.conv1x1', 'downsample.0')
                if 'downsample.bn' in k:
                    new_k = new_k.replace('downsample.bn', 'downsample.1')
                # if 'downsample.0' in k:
                #     new_k = k.replace('downsample.0', 'downsample.conv1x1')
                # if 'downsample.1' in k:
                #     new_k = k.replace('downsample.1', 'downsample.bn')
                new_state_dict[new_k] = v
            errors = self.teacher_network.load_state_dict(new_state_dict, strict=False)
            logger.warning(f'{errors}')

    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def save_parameters(self, results_folder, run_id, epoch=-1, train_type='main', end='no', prefix=''):
        if utils.is_main_process():
            path = os.path.join(results_folder, f"{prefix}net_{run_id}_task_{self._task}.pth")
            logger.info(f"Saving model at {path}.")
            save_content = {'backbone_fc': self.network.state_dict(), 'backbone_optim': self._optimizer.state_dict(),
                            'task_id': self._task, 'run_id': run_id, 'epoch': epoch,
                            'type': train_type,
                            'end': end}
            torch.save(save_content, path)

    def load_parameters(self, directory, run_id, path=None):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth") if path is None else path
        if not os.path.exists(path):
            assert FileNotFoundError, f"check your path, {path}"

        logger.info(f"Loading model at {path}.")
        load_content = torch.load(path, map_location=torch.device('cpu'))
        epoch = load_content['epoch']
        task_id = load_content['task_id']
        end = load_content.get('end', True)
        train_type = load_content.get('type', 'main')
        logger.info(f'Checkpoint status: \n'
                    f'\t| Checkpoint path | \t {directory}\n'
                    f'\t| Epoch           | \t {epoch}\n'
                    f'\t| Task ID         | \t {task_id}\n'
                    f'\t| End of task?    | \t {end}\n'
                    f'\t| training type   | \t {train_type}')
        if end.lower() == 'yes':  # bug here
            logger.info(f"The checkpoint is saved after a full stage, will start a new stage {task_id + 1}")
            resume_from_epoch = None
        else:
            logger.info(f"The checkpoint is saved in the middle of a training stage,"
                        f" will resume from epoch {epoch + 1} of stage {task_id}")
            resume_from_epoch = epoch

            # load backbone and fc
        logger.info('loading backbone fc')
        backbone_state_dict = load_content['backbone_fc']
        self.load_state_dict_force(self._network, backbone_state_dict)

        return resume_from_epoch, train_type

    @staticmethod
    def load_state_dict_force(model: nn.Module, state_dict):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            logger.warning(
                f'Loading state dict with following errors: '
                f'\n\t Missing keys:--------------\n{pprint.pformat(missing_keys)}'
                f'\n\t unexpected_keys:-----------\n{pprint.pformat(unexpected_keys)}')
        else:
            logger.info('Successfully match all keys.')

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    # ----------
    # Public API
    # ----------
    def get_groupwise_parameters(self):
        params = []
        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            groupwise_factor = self._groupwise_factors
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                logger.info(f"Group: {group_name}, lr: {self._lr * factor}.")
        return params

    def _before_task(self, train_loader, val_loader, dataset):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        if self.use_teacher_distillate and isinstance(self.teacher_network, network.BasicNet):
            self.teacher_network.add_classes(self._task_size)

        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        namespace_args = utils.lr_scaler(self.pretrain_lr_config, train_loader.batch_size)
        self.pretrain_optimizer = create_optimizer(namespace_args, self._network)
        self.pretrain_scheduler, _ = create_scheduler(namespace_args, self.pretrain_optimizer)
        if self._task == 0:
            self._n_epochs = self.pretrain_epochs

        namespace_args = utils.lr_scaler(self.args, train_loader.batch_size)
        if self._groupwise_factors:
            group_wise_params = self.get_groupwise_parameters()
            self._optimizer = create_optimizer(namespace_args, group_wise_params)
        else:
            self._optimizer = create_optimizer(namespace_args, self._network)
        self._scheduler, _ = create_scheduler(namespace_args, self._optimizer)

        # ema
        enable_ema_config = self.args.get("model_ema", None)
        ema_decay = self.args.get("model_ema_decay", False)
        model_ema_force_cpu = self.args.get("model_ema_force_cpu", False)
        if enable_ema_config is not None:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(self._network, decay=ema_decay, device='cpu' if model_ema_force_cpu else '',
                                      resume='')
        else:
            self.model_ema = None

        # if args.distributed:
        if self.ddp:
            self.model_with_ddp = torch.nn.parallel.DistributedDataParallel(self._network,
                                                                            device_ids=[self._multiple_devices],
                                                                            find_unused_parameters=True)
            self.model_without_ddp = self.model_with_ddp.module

        all_param = 0
        trainable_param = 0
        self.n_parameters = sum(p.numel() for p in self.model_with_ddp.parameters() if p.requires_grad)
        for i in self.network.named_parameters():
            all_param += i[1].numel()
            if i[1].requires_grad:
                trainable_param += i[1].numel()
        logger.info(
            f'{pprint.pformat({i[0]: i[1].requires_grad for i in self.network.named_parameters()}, sort_dicts=False)}')
        logger.info(
            f'All params: [{all_param / 1e6}M] ; Trainable params:[{trainable_param / 1e6}M], [{(trainable_param / all_param) * 100}%]')
        torch.distributed.barrier()
        self.loss_scaler = NativeScaler()

    def train_task(self, train_loader, val_loader, run_id=0, stage0=True, task_id=0, finetune=False, initial_epoch=0):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        if self.use_teacher_distillate:
            if isinstance(self.teacher_network, network.BasicNet):
                logger.info(f'Load teacher.')
                self.load_teacher_statedict(self.teacher_state_dict_path)
                logger.info(f'testing teacher')
            else:
                self.teacher_network.eval()
            evaluate_pure(val_loader, self.teacher_network, device='cuda', nb_classes=self._n_classes)
        self._training_step(train_loader, val_loader, initial_epoch, self._n_epochs,
                            optimizer=self.pretrain_optimizer if task_id == 0 else None,
                            scheduler=self.pretrain_scheduler if task_id == 0 else None,
                            )
        self.model_without_ddp = self.model_with_ddp.module
        self._network = self.model_with_ddp.module

    def _training_step(self, train_loader, val_loader, initial_epoch, nb_epochs, optimizer=None, scheduler=None,
                       run_id=0, ):
        mixup = self.args.get("mixup", 0.8)
        cutmix = self.args.get("cutmix", 1.0)
        mixup_prob = self.args.get("mixup_prob", 1)
        mixup_switch_prob = self.args.get("mixup_switch_prob", 0.5)
        cutmix_minmax = self.args.get("cutmix_minmax", None)
        mixup_mode = self.args.get("mixup_mode", "batch")
        smoothing = self.args.get("label_smoothing", 0.1)
        nb_classes = self._n_classes
        log_period = self.args.get("log_preiod", 10)  # iter
        checkpoint_eval_period = self.args.get("checkpoint_eval_period", 10)  # epoch
        mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
        if mixup_active:
            if self.local_mixup:
                mixup_fn = myMixup(
                    mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                    prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
                    label_smoothing=smoothing, num_classes=nb_classes, )
            else:
                print(f'Cutmix {cutmix}, mixup {mixup}')
                mixup_fn = Mixup(
                    mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                    prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
                    label_smoothing=smoothing, num_classes=nb_classes)

        else:
            mixup_fn = None

        test_stats = None
        max_accuracy = 0.0
        start_time = time.time()
        logger.info(f"start training for {nb_epochs - initial_epoch} epochs.")
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        _optimizer = self._optimizer if optimizer is None else optimizer
        _scheduler = self._scheduler if scheduler is None else scheduler
        criterion = SoftTargetCrossEntropy()
        for epoch in range(initial_epoch, nb_epochs):
            _scheduler.step(epoch)
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
                # self.model_with_ddp.train()
                # self.model_without_ddp.train()
            if self.teacher_network is not None:
                train_stats = train_one_epoch_baseline(
                    self.model_with_ddp,
                    train_loader,
                    _optimizer,
                    self._device,
                    epoch,
                    self.loss_scaler,
                    self.clip_grad,
                    mixup_fn,
                    set_training_mode=True,
                    print_freq=log_period,
                    nb_classes=nb_classes,
                    teacher_model=self.teacher_network,
                )
            else:
                train_stats = train_one_epoch_baseline_notea(self.model_with_ddp,
                                                             criterion,
                                                             train_loader, _optimizer, self._device, epoch,
                                                             self.loss_scaler, self.clip_grad, mixup_fn=mixup_fn,
                                                             set_training_mode=True, )

            if self._result_folder and (epoch + 1) % 5 == 0:
                checkpoint_paths = [os.path.join(self._result_folder, 'checkpoint.pth')]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': self.model_without_ddp.state_dict(),
                        'optimizer': _optimizer.state_dict(),
                        'lr_scheduler': _scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(self.model_ema),
                        'scaler': self.loss_scaler.state_dict(),
                        'args': self.args,
                    }, checkpoint_path)
            torch.distributed.barrier()
            torch.cuda.synchronize()
            if epoch % checkpoint_eval_period == checkpoint_eval_period - 1:
                test_stats = evaluate_pure(val_loader, self.model_with_ddp, device='cuda', nb_classes=nb_classes, )
                logger.info(
                    f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")

                if max_accuracy < test_stats["acc1"]:
                    max_accuracy = test_stats["acc1"]
                    if self._result_folder:
                        checkpoint_paths = [os.path.join(self._result_folder, 'checkpoint.pth')]
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': self.model_without_ddp.state_dict(),
                                'optimizer': _optimizer.state_dict(),
                                'lr_scheduler': _scheduler.state_dict(),
                                'epoch': epoch,
                                'model_ema': get_state_dict(self.model_ema),
                                'scaler': self.loss_scaler.state_dict(),
                                'args': self.args, }, checkpoint_path)

                logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            if utils.is_main_process() and (epoch + 1) % 5 == 0:
                self.save_parameters(self._result_folder, run_id=run_id, epoch=epoch)
            if test_stats is not None:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': self.n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                             'n_parameters': self.n_parameters}

            if self._result_folder and utils.is_main_process():
                with open(os.path.join(self._result_folder, 'log.txt'), 'a') as f:
                    # with (os.path.join(self._result_folder , "log.txt").open("a")) as f:
                    f.write(json.dumps(log_stats) + "\n")

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info('Training time {}'.format(total_time_str))

    def _after_task_intensive(self, inc_dataset, test_loader):
        # if self._memory_size > 0:
        #     self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
        #         inc_dataset, self._herding_indexes, extract_distributed=True
        #     )
        # else:
        self._class_means = None
        save_pretrained_path = self.args.get('save_pretrained_path', None)
        if save_pretrained_path is not None:
            save_path = os.path.join(save_pretrained_path, 'best_checkpoint.pth')
            torch.save(self.network.backbone.state_dict(), save_path)


    def update_center_lib(self, inc_dataset):
        return

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().to(self._device)
        self._network.on_task_end()
        # self.plot_tsne()

    def _eval_task(self, data_loader):
        ypreds, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)
        all_acc = accuracy_per_task(ypreds, ytrue, init_task_size=self._base_task_size, task_size=self._task_size,
                                    topk=1)
        return ypreds, ytrue

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def compute_accuracy(self, model, loader, class_means):
        if self._evaluation_type in ("icarl", "nme"):
            features, targets_ = utils.extract_features(model, loader, distributed=True, )
            features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

            # Compute score for iCaRL
            sqd = cdist(class_means, features, 'sqeuclidean')
            score_icarl = (-sqd).T
            return score_icarl, targets_
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []
            net = self._network
            net.eval()
            for input_dict in loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                with torch.no_grad():
                    logits = net(inputs)['logits'].detach().cpu()
                preds = F.softmax(logits, dim=-1)
                del logits
                ypred.append(preds.cpu().numpy())
            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def get_memory(self):
        return self._data_memory, self._targets_memory

    def set_result_folder(self, path):
        self._result_folder = path

    # def compute_accuracy(self, model, loader, class_means, APG=False):
    #     if self._evaluation_type in ("icarl", "nme"):
    #         features, targets_ = utils.extract_features(model, loader, distributed=True,
    #                                                     APG=self.adaptive_prompt_generator if APG else None)
    #
    #         features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
    #
    #         # Compute score for iCaRL
    #         sqd = cdist(class_means, features, 'sqeuclidean')
    #         score_icarl = (-sqd).T
    #         return score_icarl, targets_
    #     elif self._evaluation_type in ("softmax", "cnn"):
    #         ypred = []
    #         ytrue = []
    #         # if hasattr(self, 'model_with_ddp'):
    #         #     net = self.model_with_ddp
    #         # else:
    #         net = self._network
    #         net.eval()
    #         self.adaptive_prompt_generator.eval()
    #         for input_dict in loader:
    #             ytrue.append(input_dict["targets"].numpy())
    #
    #             inputs = input_dict["inputs"].to(self._device)
    #             with torch.no_grad():
    #                 if APG:
    #                     _, low_level_feat = net.backbone.forward_features(inputs, return_immediate_feat=True,
    #                                                                       break_immediate=True)
    #                     # low_level_feat = low_level_feat.mean(dim=1).unsqueeze(1)
    #                     _low_level_feat = low_level_feat[:, 0].unsqueeze(1)
    #                     # low_level_feat = low_level_feat.mean()
    #                     prompts = self.adaptive_prompt_generator(_low_level_feat)
    #                     logits = net(low_level_feat, extra_tokens=prompts, shortcut=True).detach().cpu()
    #                 else:
    #                     logits = net(inputs, ).detach().cpu()
    #             preds = F.softmax(logits, dim=-1)
    #             del logits
    #             ypred.append(preds.cpu().numpy())
    #         ypred = np.concatenate(ypred)
    #         ytrue = np.concatenate(ytrue)
    #
    #         self._last_results = (ypred, ytrue)
    #
    #         return ypred, ytrue
    #     else:
    #         raise ValueError(self._evaluation_type)

    # def testing(self, adaptive_prompt_generator, ddp_model, ddp_model_backbone, val_loader, ddp=True):
    #     if ddp:
    #         ddp_model_backbone = ddp_model_backbone.module
    #         ddp_model = ddp_model.module
    #
    #     test_stats = evaluate(val_loader, ddp_model_backbone, self._device, APG=None, init_task=self._base_task_size,
    #                           task_size=self._task_size)
    #     logger.info(
    #         f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
    #     adaptive_prompt_generator.train()
    #     ddp_model.train()
    #     ddp_model_backbone.train()
    #
    # def reset_fc(self):
    #     self.old_fc = copy.deepcopy(self._network.classifier.state_dict())
    #     self._network.classifier.reset_weights()


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None


class myMixup(Mixup):
    def __init__(self, *args, n_adapt=1, **kwargs):
        super(myMixup, self).__init__(*args, **kwargs)
        self.n_adapt = n_adapt

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device)
        if self.n_adapt > 1:
            bs = len(target)
            factor = self.n_adapt
            extended_classes = self.n_adapt * self.num_classes
            pad = (0, extended_classes - self.num_classes, 0, 0)
            padded = torch.nn.functional.pad(target, pad=pad, mode='constant', value=0)
            padded = padded.reshape(bs, factor, self.num_classes).transpose(-1, -2).reshape(bs, extended_classes)
        else:
            padded = None
        return x, (target, padded)
