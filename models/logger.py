import logging
import sys
import torch
import os
import time
from logging import FileHandler
import torch.distributed as dist
# def set_logging_level(logging_level):
#     logging_level = logging_level.lower()
# 
#     if logging_level == "critical":
#         level = logging.CRITICAL
#     elif logging_level == "warning":
#         level = logging.WARNING
#     elif logging_level == "info":
#         level = logging.INFO
#     else:
#         level = logging.DEBUG
# 
#     logging.basicConfig(
#         format='%(asctime)s [%(filename)s]: %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=level
#     )
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

class Master_FileHandler(FileHandler):
    def emit(self, record):
        if is_main_process():
            super().emit(record)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


class LOG:
    def __init__(self):
        handler = logging.StreamHandler()
        self.formatter = logging.Formatter(
            '[%(asctime)s] %(filename)-8s :: %(message)s',
            datefmt='%m-%d %H:%M:%S')
        handler.setFormatter(self.formatter)
        # self.LOGGER = logging.getLogger('global')
        self.LOGGER = logging.getLogger('')
        self.LOGGER.addHandler(handler)
        self.LOGGER.setLevel(logging.INFO)

    def add_file_headler(self, save_path):
        save_path = os.path.join(save_path, f'log_{time_string()}.log')
        print(f'logging path: {save_path}')
        if is_main_process():
            f = open(save_path, 'w')
            f.close()
        # fhandler = Master_FileHandler(save_path, mode='w')
        fhandler = FileHandler(save_path, mode='w')
        fhandler.setFormatter(self.formatter)
        self.LOGGER.addHandler(fhandler)

    def print_baisic_info(self):
        self.LOGGER.info("python version : {}".format(sys.version.replace('\n', ' ')))
        self.LOGGER.info("torch  version : {}".format(torch.__version__))


LOGGER = LOG()

if __name__ == '__main__':
    import random

    log = LOG()
    log.add_file_headler('/home/yuming/')
    log.LOGGER.info(f'info message{random.random()}')
    log.LOGGER.warning('warning message')
    log.LOGGER.error('error message')
    log.print_baisic_info()
