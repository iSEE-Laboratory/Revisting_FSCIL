import torch

from models.base.fscil_trainer import FSCILTrainer as Trainer
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from torch.distributions.multivariate_normal import MultivariateNormal
from .Network import MYNET
from models.logger import LOGGER
from models.metric import get_gacc

log = LOGGER.LOGGER


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        LOGGER.add_file_headler(self.args.save_path)
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            log.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            log.info('random init params')
            if args.start_session > 0:
                log.info('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader, trainloader_pk = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
            trainloader_pk = None
        return trainset, trainloader, testloader, trainloader_pk

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]
        all_feat_distributions = {}
        hyper_params = {'cos': args.fr_cos, 'rkd': args.fr_rkd, 'ce_novel': args.fr_ce_novel,
                        'ce_current': args.fr_ce_current, 'ce_global': args.fr_ce_global,
                        'kl': args.fr_kl, 'rkd_inter': args.rkd_inter, 'rkd_intra': args.rkd_intra,
                        'rkd_split': args.rkd_split, 'extra_rkd_split': args.extra_rkd_split,
                        'rkd_intra_extra': args.rkd_intra_extra, 'rkd_inter_extra': args.rkd_inter_extra,
                        }

        all_acc = []
        if args.start_session > 0:  # need to test
            train_set, trainloader, testloader, trainloader_pk = self.get_dataloader(0)  # get session 0 dataloaders

            test(self.model.module.encoder, self.model.module.fc, testloader, 0, args, 0, )
            trainloader.dataset.transform = testloader.dataset.transform
            features, features_all, labels = extract_features(self.model.module.encoder, trainloader, args)
            if args.re_extract_avg:
                self.model.module.update_fc_avg(self.model.module.fc, features, features_all, labels, np.unique(labels),
                                                args.output_blocks)
                # [])
                test(self.model.module.encoder, self.model.module.fc, testloader, 0, args, 0, )

            for cls in np.unique(labels):
                cls_mask = labels == cls
                cls_feats = features[cls_mask]
                feat_mean = cls_feats.mean(0)
                feat_cov = torch.cov(cls_feats.T)
                tmp = torch.eye(len(feat_cov)) * args.eps
                distribution_feat = MultivariateNormal(loc=feat_mean, covariance_matrix=feat_cov + tmp)

                all_feat_distributions[cls.item()] = {'strong_distribution': distribution_feat,
                                                      'strong_mean': feat_mean,
                                                      'strong_cov': feat_cov, }
                for i in args.output_blocks:
                    cur_features = features_all[f'layer{i}']
                    cls_feats = cur_features[cls_mask]
                    feat_mean = cls_feats.mean(0)
                    feat_cov = torch.cov(cls_feats.T)
                    tmp = torch.eye(len(feat_cov)) * args.eps
                    distribution_feat = MultivariateNormal(loc=feat_mean, covariance_matrix=feat_cov + tmp)

                    all_feat_distributions[cls.item()].update({f'layer{i}_distribution': distribution_feat,
                                                               f'layer{i}_mean': feat_mean,
                                                               f'layer{i}_cov': feat_cov, })

            base_classes = torch.unique(labels)

            # debias module
            if args.resume_fr is not None:
                log.info(f'resume from {args.resume_fr}')
                (chkpt, start_session) = torch.load(args.resume_fr, 'cpu')
                err = self.model.module.feature_rectification.load_state_dict(chkpt, strict=False)
                log.warn(f'err: {err}')
                self.best_model_dict = deepcopy(self.model.state_dict())
            else:
                if args.base_fr:
                    save_model_dir = os.path.join(args.save_path, 'session' + str(0) + '_feat_rect.pth')
                    train_feature_rectification_base(self.model.module.encoder,
                                                     self.model.module.feature_rectification,
                                                     self.model.module.fc,
                                                     args,
                                                     trainloader_pk,
                                                     testloader,
                                                     hyper_params,
                                                     testing_interval=10,
                                                     session=0, )
                    all_acc_per_task, all_acc_per_task_list = test_rectification(self.model.module.encoder,
                                                                                 self.model.module.feature_rectification,
                                                                                 self.model.module.fc, testloader, -1,
                                                                                 args,
                                                                                 0)
                    all_acc.append(all_acc_per_task_list)
                    g_acc, area = get_gacc(ratio=int(args.base_class / args.way),
                                           all_acc=all_acc)
                    log.info(f'Task [{0}]\tgAcc: {g_acc}, gAcc_area: {area}')
                    torch.save((self.model.module.feature_rectification.state_dict(), 1), save_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                start_session = None

        else:
            assert False
            # all_feat_distributions = None
            # base_classes = None
        start_session = start_session if start_session is not None else args.start_session
        for session in range(start_session, args.sessions):

            train_set, trainloader, testloader, _ = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict, strict=False)

            if session == 0:
                raise NotImplementedError
            else:  # incremental learning sessions
                log.info("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()

                trainloader_new = deepcopy(trainloader)
                trainloader_new.dataset.transform = testloader.dataset.transform
                # get novel data
                assert len(trainloader_new) == 1  # assume all data are loaded in a batch (in few shot learning stage)
                for batch in trainloader_new:
                    data, label = [_.cuda() for _ in batch]
                with torch.no_grad():
                    all_feat_distributions = get_MVNs(data, label,
                                                      base_label=base_classes,
                                                      encoder=self.model.module.encoder,
                                                      output_blocks=args.output_blocks,
                                                      eps=args.eps,
                                                      MVN_distributions=all_feat_distributions,
                                                      k=args.K)

                self.model.module.update_fc(dataloader=trainloader_new,
                                            encoder=self.model.module.encoder,
                                            output_blocks=args.output_blocks)

                # test via vanilla networks
                test(self.model.module.encoder, self.model.module.fc, testloader, 0, args, session, )

                log.info('-' * 50 + '\n')
                train_feature_rectification_distribution(self.model.module.encoder,
                                                         self.model.module.feature_rectification,
                                                         self.model.module.fc,
                                                         args,
                                                         all_feat_distributions,
                                                         testloader,
                                                         hyper_params=hyper_params,
                                                         testing_interval=10,
                                                         session=session,
                                                         dataloader=trainloader_new)
                all_acc_per_task, all_acc_per_task_list = test_rectification(self.model.module.encoder,
                                                                             self.model.module.feature_rectification,
                                                                             self.model.module.fc, testloader, -1, args,
                                                                             session)
                all_acc.append(all_acc_per_task_list)
                g_acc, area = get_gacc(ratio=int(args.base_class / args.way),
                                       all_acc=all_acc)
                log.info(f'Task [{session}]\tgAcc: {g_acc}, gAcc_area: {area}')
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                log.info('Saving model to :%s' % save_model_dir)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        log.info('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        self.args.save_path = self.args.save_path + self.args.exp_name
        ensure_path(self.args.save_path)
        return None
