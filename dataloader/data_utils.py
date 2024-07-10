import numpy as np
import torch
from dataloader.sampler import CategoriesSampler


def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    args.Dataset = Dataset
    return args


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader, trainloader_pk = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
        trainloader_pk = None
    return trainset, trainloader, testloader, trainloader_pk


def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True, img_size=args.image_size)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True, img_size=args.image_size)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, size=args.image_size)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, size=args.image_size)

    sampler = PKsampler(trainset, p=args.p, k=args.k)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    #
    trainloader_pk = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, sampler=sampler,
                                              num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader, trainloader_pk


class PKsampler(torch.utils.data.Sampler):
    def __init__(self, dataset, p=64, k=8):
        self.p = p  # num samples
        self.k = k  # num classes
        self.dataset = dataset
        # self.num_classes = np.unique(self.dataset.targets).shape[0]
        self.all_classes = np.unique(self.dataset.targets)
        self.all_targets = self.dataset.targets
        self.batch_size = self.p * self.k

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # choose target classes randomly
        num_batches = len(self.dataset) // self.batch_size
        res = []
        for i in range(num_batches):
            target_classes = np.random.choice(self.all_classes, self.k)
            for cls in target_classes:
                # choose samples
                cls_idx = np.where(self.all_targets == cls)[0]
                cls_idx = np.random.choice(cls_idx, self.p).tolist()
                res.append(cls_idx)
        res = np.concatenate(res).tolist()
        return iter(res)


def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path, size=args.image_size)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index, size=args.image_size)

    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args, session):
    # txt_path = "data/index_list/" + args.dataset + f"/new_split_seed{args.dataset_seed}" + \
    #            "/session_" + str(session + 1) + '.txt'
    if args.dataset_seed is not None:
        txt_path = "data/index_list/" + args.dataset + f"/new_split_seed{args.dataset_seed}" + \
                   "/session_" + str(session + 1) + '.txt'
    else:
        txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    print(f'dataset seed {args.dataset_seed}, loading from {txt_path}')
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, img_size=args.image_size)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path, size=args.image_size)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers_new, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers_new, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False, img_size=args.image_size)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_new, size=args.image_size)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers_new, pin_memory=True)

    return trainset, trainloader, testloader


def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list
