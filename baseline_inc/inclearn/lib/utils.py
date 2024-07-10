import datetime
import logging
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import time
from collections import defaultdict, deque
import argparse
from tqdm import tqdm
# logger = logging.getLogger(__name__)
from inclearn.lib.logger import LOGGER
from timm.utils.clip_grad import dispatch_clip_grad

logger = LOGGER.LOGGER


def to_onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def to_smooth_labeling(targets, n_classes, label_num=1, eps=0.3, share=True):
    smooth_label = torch.ones(targets.shape[0], n_classes).to(targets.device) * (eps / (n_classes - label_num))
    v = (1 - eps) / label_num if share else (1 - eps)
    for i in range(label_num):
        smooth_label.scatter_(dim=1, index=targets[:, i].view(-1, 1), value=v)
    return smooth_label


def check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.).item())


def compute_accuracy(ypred, ytrue, task_size=10):
    all_acc = {}

    all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    for class_id in range(0, np.max(ytrue), task_size):
        idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

        label = "{}-{}".format(
            str(class_id).rjust(2, "0"),
            str(class_id + task_size - 1).rjust(2, "0")
        )
        all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def extract_features(model, loader, distributed=False, additional_parameters=None, return_numpy=True):
    def to_numpy(x):
        if return_numpy:
            return x.numpy()
        else:
            return x

    targets, features = [], torch.tensor([])
    state = model.training
    model.eval()
    additional_parameters = {} if additional_parameters is None else additional_parameters

    num_ranks = get_world_size()
    for input_dict in tqdm(loader):
        inputs, _targets = input_dict["inputs"], input_dict["targets"]
        if distributed:
            _targets = _targets.cuda(non_blocking=True)
            all_target_list = [torch.zeros_like(_targets, dtype=torch.long) for _ in
                               range(num_ranks)]
            dist.all_gather(all_target_list, _targets)
            _targets = to_numpy(torch.cat(all_target_list, dim=0).cpu())
            del all_target_list
            torch.distributed.barrier()
            _features = model.extract(inputs.cuda(non_blocking=True), **additional_parameters)
            _features = _features.contiguous()
            all_feat_list = [torch.zeros_like(_features, dtype=torch.float32) for _ in
                             range(num_ranks)]
            dist.all_gather(all_feat_list, _features)
            _features = to_numpy(torch.cat(all_feat_list, dim=0).cpu())

            del all_feat_list

        else:
            _features = to_numpy(
                model.extract(inputs.to(model.device), **additional_parameters).detach().cpu())
            _targets = to_numpy(_targets)

        features = torch.cat((features, _features), dim=0)
        targets.append(_targets)

    logger.info('finished extract features.')
    torch.distributed.barrier()
    torch.cuda.synchronize()
    model.train(state)

    if not return_numpy:
        return features, torch.cat(targets)
    return np.concatenate(features), np.concatenate(targets)


def extract_features_immediate(model, loader, distributed=False):
    targets, features = [], []

    state = model.training
    model.eval()

    num_ranks = get_world_size()
    for input_dict in tqdm(loader):
        inputs, _targets = input_dict["inputs"], input_dict["targets"]
        if distributed:
            _targets = _targets.cuda(non_blocking=True)
            all_target_list = [torch.zeros_like(_targets, dtype=torch.long) for _ in
                               range(num_ranks)]
            dist.all_gather(all_target_list, _targets)
            _targets = torch.cat(all_target_list, dim=0).cpu().numpy()
            del all_target_list
            torch.distributed.barrier()
            _features = model.extract(inputs.cuda(non_blocking=True), return_immediate=True).detach().contiguous()
            all_feat_list = [torch.zeros_like(_features, dtype=torch.float32) for _ in range(num_ranks)]
            dist.all_gather(all_feat_list, _features)
            _features = torch.cat(all_feat_list, dim=0).cpu().numpy()
            del all_feat_list

        else:
            _features = model.extract(inputs.to(model.device), return_immediate=True).detach().cpu().numpy()
            _targets = _targets.numpy()

        features.append(_features)
        targets.append(_targets)

    model.train(state)
    return np.concatenate(features), np.concatenate(targets)


def compute_centroids(model, loader):
    features, targets = extract_features(model, loader)

    centroids_features, centroids_targets = [], []
    for t in np.unique(targets):
        indexes = np.where(targets == t)[0]

        centroids_features.append(np.mean(features[indexes], axis=0, keepdims=True))
        centroids_targets.append(t)

    return np.concatenate(centroids_features), np.array(centroids_targets)


def classify(model, loader):
    targets, predictions = [], []

    for input_dict in loader:
        inputs, _targets = input_dict["inputs"], input_dict["targets"]

        outputs = model(inputs.to(model.device))
        if not isinstance(outputs, list):
            outputs = [outputs]

        preds = outputs[-1].argmax(dim=1).detach().cpu().numpy()

        predictions.append(preds)
        targets.append(_targets)

    return np.concatenate(predictions), np.concatenate(targets)


def plot_tsne(path, embeddings, targets):
    assert embeddings.shape[0] == targets.shape[0]

    tsne = manifold.TSNE(n_components=2)

    embeddings_2d = tsne.fit_transform(embeddings)
    plt.scatter(
        embeddings_2d[..., 0],
        embeddings_2d[..., 1],
        c=targets,
        vmin=min(targets),
        vmax=max(targets),
        s=10,
        cmap=mpl.cm.get_cmap('RdYlBu')
    )

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)


def add_new_weights(network, weight_generation, current_nb_classes, task_size, inc_dataset):
    if isinstance(weight_generation, str):
        warnings.warn("Use a dict for weight_generation instead of str", DeprecationWarning)
        weight_generation = {"type": weight_generation}

    if weight_generation["type"] == "imprinted":
        logger.info("Generating imprinted weights")

        network.add_imprinted_classes(
            list(range(current_nb_classes, current_nb_classes + task_size)), inc_dataset,
            **weight_generation
        )
    elif weight_generation["type"] == "embedding":
        logger.info("Generating embedding weights")

        mean_embeddings = []
        for class_index in range(current_nb_classes, current_nb_classes + task_size):
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = extract_features(network, loader)
            features = features / np.linalg.norm(features, axis=-1)[..., None]

            mean = np.mean(features, axis=0)
            if weight_generation.get("proxy_per_class", 1) == 1:
                mean_embeddings.append(mean)
            else:
                std = np.std(features, axis=0, ddof=1)
                mean_embeddings.extend(
                    [
                        np.random.normal(loc=mean, scale=std)
                        for _ in range(weight_generation.get("proxy_per_class", 1))
                    ]
                )

        network.add_custom_weights(np.stack(mean_embeddings))
    elif weight_generation["type"] == "basic":
        network.add_classes(task_size)
    elif weight_generation["type"] == "ghosts":
        features, targets = weight_generation["ghosts"]
        features = features.cpu().numpy()
        targets = targets.cpu().numpy()

        weights = []
        for class_id in range(current_nb_classes, current_nb_classes + task_size):
            indexes = np.where(targets == class_id)[0]

            class_features = features[indexes]
            if len(class_features) == 0:
                raise Exception(f"No ghost class_id={class_id} for weight generation!")
            weights.append(np.mean(class_features, axis=0))

        weights = torch.tensor(np.stack(weights)).float()
        network.add_custom_weights(weights, ponderate=weight_generation.get("ponderate"))
    else:
        raise ValueError("Unknown weight generation type {}.".format(weight_generation["type"]))


def apply_kmeans(features, targets, nb_clusters, pre_normalization):
    logger.info(
        "Kmeans on {} samples (pre-normalized: {}) with {} clusters per class".format(
            len(features), pre_normalization, nb_clusters
        )
    )

    new_features = []
    new_targets = []
    for class_index in np.unique(targets):
        kmeans = KMeans(n_clusters=nb_clusters)

        class_sample_indexes = np.where(targets == class_index)[0]
        class_features = features[class_sample_indexes]
        class_targets = np.ones((nb_clusters,)) * class_index

        if pre_normalization:
            class_features = class_features / np.linalg.norm(class_features, axis=-1).reshape(-1, 1)

        kmeans.fit(class_features)
        new_features.append(kmeans.cluster_centers_)
        new_targets.append(class_targets)

    return np.concatenate(new_features), np.concatenate(new_targets)


def apply_knn(
        features,
        targets,
        features_test,
        targets_test,
        nb_neighbors,
        normalize=True,
        weights="uniform"
):
    logger.info(
        "KNN with {} neighbors and pre-normalized features: {}, weights: {}.".format(
            nb_neighbors, normalize, weights
        )
    )

    if normalize:
        features = features / np.linalg.norm(features, axis=-1).reshape(-1, 1)

    knn = KNeighborsClassifier(n_neighbors=nb_neighbors, n_jobs=10, weights=weights)
    knn.fit(features, targets)

    if normalize:
        features_test = features_test / np.linalg.norm(features_test, axis=-1).reshape(-1, 1)

    pred_targets = knn.predict(features_test)

    return pred_targets, targets_test


def select_class_samples(samples, targets, selected_class):
    indexes = np.where(targets == selected_class)[0]
    return samples[indexes], targets[indexes]


def matrix_infinity_norm(matrix):
    # Matrix is of shape (w, h)
    matrix = torch.abs(matrix)

    summed_col = matrix.sum(1)  # Shape (w,)
    return torch.max(summed_col)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args['rank'] = int(os.environ["RANK"])
        args['world_size'] = int(os.environ['WORLD_SIZE'])
        args['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args['rank'] = int(os.environ['SLURM_PROCID'])
        args['gpu'] = args['rank'] % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args['distributed'] = False
        return

    args['distributed'] = True

    torch.cuda.set_device(args['gpu'])
    args['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args['rank'], args['dist_url']), flush=True)
    torch.distributed.init_process_group(backend=args['dist_backend'], init_method=args['dist_url'],
                                         world_size=args['world_size'], rank=args['rank'])
    torch.distributed.barrier()
    debug = args.get('debug', False)
    setup_for_distributed(args['rank'] == 0, debug=debug)


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


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master, debug=False):
    """
    This function disables printing when not in master process
    """
    if debug:
        return
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    import sys
    stderr_write = sys.stderr.write
    stdout_write = sys.stdout.write

    def e_write(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            return stderr_write(*args, **kwargs)
        # return 0

    def o_write(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            return stdout_write(*args, **kwargs)
        # return 0

    sys.stderr.write = e_write
    sys.stdout.write = o_write


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def extract_transform_args(attr_name_list, attr_ori_dict, namespace=True):
    target_dict = {i: attr_ori_dict[i] for i in attr_name_list}
    if namespace:
        return argparse.Namespace(**target_dict)
    else:
        return target_dict


class MyScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def loss_backward(self, loss, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)

    def __call__(self, optimizer, clip_grad=None, clip_mode='norm', parameters=None):
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def lr_scaler(lr_config, batch_size, convert=True):
    if convert:
        lr_config['scaled'] = False
        namespace_args = argparse.Namespace(**lr_config)  # to co-operate with timm, convert dict into Namespace.
    else:
        namespace_args = lr_config
    if namespace_args.scaled:
        logger.warning(f'Namespace has scaled once, skipped the scaled process.')
        return namespace_args
    linear_scaled_lr = namespace_args.lr * batch_size * get_world_size() / 1024.0
    linear_scaled_warmup_lr = namespace_args.warmup_lr * batch_size * get_world_size() / 1024.0
    linear_scaled_minlr = namespace_args.min_lr * batch_size * get_world_size() / 1024.0
    logger.info(f'scaled lr: \n'
                f'\t Origin lr {namespace_args.lr} => new lr {linear_scaled_lr}\n'
                f'\t Origin warmup lr {namespace_args.warmup_lr} => new warmup lr {linear_scaled_warmup_lr}\n'
                f'\t Origin min lr {namespace_args.min_lr} => new min lr {linear_scaled_minlr}\n')
    if hasattr(namespace_args, 'lr_cls'):
        info = f'\t Origin cls lr {namespace_args.lr_cls}'
        namespace_args.lr_cls = namespace_args.lr_cls * batch_size * get_world_size() / 1024.0
        logger.info(f'{info} => new cls lr {namespace_args.lr_cls}')
    namespace_args.lr = linear_scaled_lr
    namespace_args.warmup_lr = linear_scaled_warmup_lr
    namespace_args.min_lr = linear_scaled_minlr
    namespace_args.scaled = True
    return namespace_args


if __name__ == '__main__':
    targets = torch.tensor(
        [
            [0, 1], [2, 3], [0, 1], [4, 5], [0, 1], [8, 9], [6, 7], [8, 9], [4, 7]
        ]
    )
    smoothing = to_smooth_labeling(targets, 10, 2, )
    pass
