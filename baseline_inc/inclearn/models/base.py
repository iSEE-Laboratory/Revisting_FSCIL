import abc
import logging
import os
import pprint

import torch

# LOGGER = logging.Logger("IncLearn", level="INFO")

# logger = logging.getLogger(__name__)
from inclearn.lib.logger import LOGGER

LOGGER = LOGGER.LOGGER
logger = LOGGER


class IncrementalLearner(abc.ABC):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """

    def __init__(self, *args, **kwargs):
        self._network = None

    def set_task_info(self, task_info):
        self._task = task_info["task"]
        self._total_n_classes = task_info["total_n_classes"]
        self._task_size = task_info["increment"]
        if self._task == 0:
            self._base_task_size = self._task_size
        self._n_train_data = task_info["n_train_data"]
        self._n_test_data = task_info["n_test_data"]
        self._n_tasks = task_info["max_task"]
        logger.info(f'_task: [{self._task}]\n'
                    f'_total_n_classes : [{self._total_n_classes}]\n'
                    f'_task_size: [{self._task_size}]\n'
                    f'_base_task_size: [{self._base_task_size}]\n'
                    f'_n_train_data: [{self._n_train_data}]\n'
                    f'_n_test_data: [{self._n_test_data}]\n'
                    f'_n_tasks: [{self._n_tasks}]')

    def before_task(self, train_loader, val_loader, dataset):
        logger.info("Before task")
        self.eval()
        self._before_task(train_loader, val_loader, dataset)

    def train_task(self, train_loader, val_loader):
        logger.info("train task")
        self.train()
        self._train_task(train_loader, val_loader)

    def after_task_intensive(self, inc_dataset, test_loader):
        logger.info("after task")
        self.eval()
        self._after_task_intensive(inc_dataset, test_loader)

    def after_task(self, inc_dataset):
        logger.info("after task")
        self.eval()
        self._after_task(inc_dataset)

    def eval_task(self, data_loader):
        logger.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory(self):
        return None

    def get_val_memory(self):
        return None

    def _before_task(self, data_loader, val_loader):
        pass

    def _train_task(self, train_loader, val_loader):
        raise NotImplementedError

    def _after_task_intensive(self, data_loader, test_loader):
        pass

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def save_metadata(self, directory, run_id):
        pass

    def load_metadata(self, directory, run_id):
        pass

    @property
    def _new_task_index(self):
        return self._task * self._task_size

    @property
    def inc_dataset(self):
        return self.__inc_dataset

    @inc_dataset.setter
    def inc_dataset(self, inc_dataset):
        self.__inc_dataset = inc_dataset

    @property
    def network(self):
        return self._network

    def save_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        logger.info(f"Saving model at {path}.")
        torch.save(self.network.state_dict(), path)

    def load_parameters(self, directory, run_id, *args, **kwargs):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        if not os.path.exists(path):
            assert FileNotFoundError, f"check your path, {path}"

        logger.info(f"Loading model at {path}.")
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = self.network.load_state_dict(state_dict, strict=False)
            if len(missing_keys) or len(unexpected_keys):
                logger.warning(
                    f'Loading state dict with following errors: '
                    f'\n\t Missing keys:--------------\n{pprint.pformat(missing_keys)}'
                    f'\n\t unexpected_keys:-----------\n{pprint.pformat(unexpected_keys)}')
            else:
                logger.info('Successfully match all keys.')

        except Exception as e:
            logger.error(e)
            logger.warning("Old method to save weights, it's deprecated!")
            self._network = torch.load(path)

    def eval(self):
        self._network.eval()
        ddp_stuff = ['model_with_ddp', 'model_without_ddp']
        for s in ddp_stuff:
            if hasattr(self, s):
                getattr(self, s).eval()

    def train(self):
        self._network.train()
        ddp_stuff = ['model_with_ddp', 'model_without_ddp']
        for s in ddp_stuff:
            if hasattr(self, s):
                getattr(self, s).train()
