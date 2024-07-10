import random
import torch
import os
import time

import numpy as np
import pprint as pprint

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label, branch_num=1, origin_bs=0):
    if branch_num == 1:
        # pred = torch.argmax(logits, dim=1)
        # return (pred == label).float().mean().item()
        return logits
    else:
        predicts = 0
        for i in range(branch_num):
            predicts += logits[i * origin_bs: (i + 1) * origin_bs, i::branch_num] / branch_num
        # predicts = torch.argmax(predicts, dim=1)
        # label = label.reshape(branch_num, -1)[0]
        # return (predicts == label).float().mean().item()
        return predicts


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def accuracy(output, targets, topk=1):
    """Computes the precision@k for the specified values of k"""
    output, targets = torch.tensor(output), torch.tensor(targets)

    batch_size = targets.shape[0]
    if batch_size == 0:
        return 0.
    nb_classes = len(np.unique(targets))
    topk = min(topk, nb_classes)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0).item()
    return round(correct_k / batch_size, 3)


def accuracy_per_task(ypreds, ytrue, init_task_size=0, task_size=10, topk=1):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    all_acc = {}
    all_acc_list = []

    all_acc["total"] = accuracy(ypreds, ytrue, topk=topk)

    if task_size is not None:

        if init_task_size > 0:
            idxes = np.where(np.logical_and(ytrue >= 0, ytrue < init_task_size))[0]

            label = "{}-{}".format(
                str(0).rjust(2, "0"),
                str(init_task_size - 1).rjust(2, "0")
            )
            acc = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)
            all_acc[label] = acc
            all_acc_list.append(acc * 100)

        for class_id in range(init_task_size, np.max(ytrue) + task_size, task_size):
            if class_id > np.max(ytrue):
                break

            idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

            label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
            )
            acc = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)
            all_acc[label] = acc
            all_acc_list.append(acc * 100)

    return all_acc, all_acc_list
