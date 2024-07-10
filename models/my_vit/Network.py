import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deit import deit_my_small_patch3_MultiLyaerOutput, \
    deit_my_small_patch8_MultiLyaerOutput, deit_my_small_patch16_MultiLyaerOutput_224
from models.myModules import FeatureRectify
import numpy as np
from models.logger import LOGGER

log = LOGGER.LOGGER


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        self.branches = 1
        self.stage0_chkpt_path = args.stage0_chkpt
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = deit_my_small_patch3_MultiLyaerOutput(num_classes=0, output_blocks=args.output_blocks,
                                                                 image_size=args.image_size)
            chkpt = torch.load(self.stage0_chkpt_path, 'cpu')['backbone_fc']
            chkpt = {i.replace('backbone.', ''): v for i, v in chkpt.items()}
            err = self.encoder.load_state_dict(chkpt, strict=False)
            log.info(err)

            self.num_features = self.encoder.num_features
        if self.args.dataset in ['mini_imagenet']:
            # self.encoder = resnet18(False, args)  # pretrained=False
            self.encoder = deit_my_small_patch8_MultiLyaerOutput(num_classes=0, output_blocks=args.output_blocks,
                                                                 image_size=args.image_size)

        elif self.args.dataset == 'cub200':
            self.encoder = self.encoder = deit_my_small_patch16_MultiLyaerOutput_224(num_classes=0, output_blocks=args.output_blocks,
                                                                 image_size=args.image_size)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790

        chkpt = torch.load(self.stage0_chkpt_path, 'cpu')['backbone_fc']
        chkpt = {i.replace('backbone.', ''): v for i, v in chkpt.items()}
        err = self.encoder.load_state_dict(chkpt, strict=False)
        log.info(err)
        self.num_features = self.encoder.num_features


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(self.num_features, self.args.num_classes,
        #                     bias=False)  # use single classifier for each branches
        self.fc = nn.ModuleDict({f'layer{i}': nn.Linear(self.num_features, self.args.num_classes, bias=False) for i in
                                 self.args.output_blocks})  # use classifier for each blocks
        self.fc['final'] = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        chkpt = torch.load(self.stage0_chkpt_path, 'cpu')['backbone_fc']  # now cosine classifier only
        chkpt_cls_weight = chkpt['classifier._weights.0'][:self.args.base_class, :]
        for i in self.fc.keys():
            self.fc[i].weight.data[:len(chkpt_cls_weight), :].copy_(chkpt_cls_weight)

        log.info('loaded fc weights and bias')

        self.feature_rectification = nn.ModuleDict({f'layer{i}': FeatureRectify(self.encoder.num_features) for i in
                                                    self.args.output_blocks})

    def forward(self, x, fc_type='fc', return_all_feats=False):
        x = self.encoder(x, return_all=True)
        all_feats = x['output_features_per_block']
        x = x['features']
        if self.mode != 'encoder':
            assert False
            # fc = getattr(self, fc_type)
            # fc_weight = fc.weight.data
            # out_dim, in_dim = fc_weight.shape
            # expanded_weights = fc_weight.repeat_interleave(self.branches, dim=0).reshape(-1, in_dim)
            # logits = torch.mm(F.normalize(x, p=2, dim=-1),
            #                   F.normalize(expanded_weights, p=2, dim=-1).T)
            # return logits
        elif self.mode == 'encoder':
            if return_all_feats:
                return x, all_feats
            else:
                return x
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, encoder, output_blocks):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            with torch.no_grad():
                strong_feat, all_feats = encoder(data, return_all=True)

        self.update_fc_avg(self.fc, strong_feat, all_feats, label, np.unique(label.detach().cpu()), output_blocks)

    def update_fc_avg(self, fc, strong_feature, all_feature, label, class_list, output_blocks):
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = strong_feature[data_index]
            proto = embedding.mean(0)
            proto = F.normalize(proto, p=2, dim=-1)
            fc['final'].weight.data[class_index] = proto

            for ii in output_blocks:
                embedding = all_feature[f'layer{ii}'][data_index]
                proto = embedding.mean(0)
                proto = F.normalize(proto, p=2, dim=-1)
                fc[f'layer{ii}'].weight.data[class_index] = proto

    def update_fc_ft(self, new_fc, data, label, session, branches=1, MVN_distributions=None, dataloader=None):
        return
