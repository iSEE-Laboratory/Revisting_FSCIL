import torch
import torch.nn as nn
import torch.nn.functional as F
from inclearn.lib.logger import LOGGER
from inclearn.backbones.etf_resnet.ETFHead import ETFHead
from inclearn.backbones.etf_resnet.mlp_ffn_neck import MLPFFNNeckTrans
from timm.models.layers.helpers import to_2tuple

logger = LOGGER.LOGGER


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])
        self.norm = nn.LayerNorm(hidden_features, eps=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.norm(x)
        return x


class ExtraMlp(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_mlps, residual=True):
        super(ExtraMlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_mlps = num_mlps
        self.residual = residual
        self.MLPs = nn.ModuleList([Mlp(in_features=self.input_dim,
                                       out_features=self.output_dim,
                                       hidden_features=self.hidden_dim)
                                   for _ in range(self.num_mlps)])

    def forward(self, x):
        for i in range(self.num_mlps):
            x_res = x
            x = self.MLPs[i](x)
            if self.residual:
                x += x_res
        return x


class TransClassifiers(nn.Module):
    classifier_type = "fc"

    def __init__(self, features_dim, device, *, proxy_per_class=1, use_bias=True, normalize=False, init="kaiming",
                 ddp=False, pre_MLP=False, MLP_num=0, **kwargs):
        super().__init__()

        self.features_dim = features_dim
        self.use_bias = use_bias
        self.init_method = init
        self.device = device
        self.normalize = normalize
        self._weights = nn.ParameterList([])
        self._bias = nn.ParameterList([]) if self.use_bias else None

        self.proxy_per_class = proxy_per_class
        logger.info(f'FC proxy :{self.proxy_per_class}')

        self.n_classes = 0
        self.ddp = ddp
        self.pre_MLP = pre_MLP
        if pre_MLP:
            logger.info('Use Pre-MLPs before the classifier!')
        self.MLP_num = MLP_num
        if self.pre_MLP:
            self.MLPs = ExtraMlp(features_dim, features_dim, features_dim, MLP_num, residual=True)
        else:
            self.MLPs = None
        # self.MLPs_enable = nn.Identity()
        self.MLPs_enable = False  # init, False

    def enable_mlp_fc(self):
        if self.MLPs_enable:
            logger.warning(f'MLPs_enable is already [True].')
        else:
            logger.info(f'Using Extra MLPs to classify.')
            assert self.pre_MLP
            self.MLPs_enable = True

    def disable_mlp_fc(self):
        if not self.MLPs_enable:
            logger.warning(f'MLPs_enable is already [False].')
        else:
            logger.info(f'NOT Using Extra MLPs to classify.')
            assert self.pre_MLP
            self.MLPs_enable = False

    def mlp_eval(self):
        logger.info('set classifier MLP to eval mode.')
        self.MLPs.eval()

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    @property
    def weights(self):
        return torch.cat([w for w in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    @property
    def bias(self):
        if self._bias is not None:
            return torch.cat([b for b in self._bias])
        return None

    @property
    def new_bias(self):
        return self._bias[-1]

    @property
    def old_bias(self):
        if len(self._bias) > 1:
            return self._bias[:-1]
        return None

    def forward_last_FC(self, features):
        if len(self._weights) == 0:
            raise Exception("Add some classes before training.")

        weights = self.weights

        if self.normalize:
            features = F.normalize(features, dim=1, p=2)

        logits = F.linear(features, weights, bias=self.bias)
        if self.proxy_per_class > 1:
            batch_size = logits.shape[0]
            nb_cls = int(logits.shape[1] / self.proxy_per_class)
            logits = logits.reshape(batch_size, nb_cls, self.proxy_per_class)
            att = F.softmax(logits, dim=-1)
            logits = (logits * att).sum(-1)

        return logits

    def forward(self, features, return_immediate_feature=False, only_MLP_features=False):
        if len(self._weights) == 0:
            raise Exception("Add some classes before training.")

        weights = self.weights
        if self.MLPs_enable:
            features = self.MLPs(features)
        if only_MLP_features:
            return features

        if self.normalize:
            features = F.normalize(features, dim=1, p=2)

        logits = F.linear(features, weights, bias=self.bias)
        if self.proxy_per_class > 1:
            batch_size = logits.shape[0]
            nb_cls = int(logits.shape[1] / self.proxy_per_class)
            logits = logits.reshape(batch_size, nb_cls, self.proxy_per_class)
            att = F.softmax(logits, dim=-1)
            logits = (logits * att).sum(-1)

        if return_immediate_feature:
            return logits, features
        else:
            return logits

    def add_classes(self, n_classes):
        self._weights.append(nn.Parameter(torch.randn(self.proxy_per_class * n_classes, self.features_dim)))
        self._init(self.init_method, self.new_weights)

        if self.use_bias:
            self._bias.append(nn.Parameter(torch.randn(n_classes * self.proxy_per_class)))
            self._init(0., self.new_bias)

        if self.ddp:
            self.to(torch.device('cuda'))
        else:
            self.to(self.device)

    def reset_weights(self):
        for w in self._weights:
            self._init(self.init_method, w)
        if self.pre_MLP:
            self.MLPs.apply(self.MLP_init)
        pass

    @staticmethod
    def MLP_init(m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(1.)
            if m.bias is not None:
                m.bias.data.fill_(0.)

    @staticmethod
    def _init(init_method, parameters):
        if isinstance(init_method, float) or isinstance(init_method, int):
            nn.init.constant_(parameters, init_method)
        elif init_method == "kaiming":
            nn.init.kaiming_normal_(parameters, nonlinearity="linear")
        else:
            raise NotImplementedError("Unknown initialization method: {}.".format(init_method))

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        if self.ddp:
            self.to(torch.device('cuda'))
        else:
            self.to(self.device)

if __name__ == '__main__':
    a = TransClassifiers(256, 'cpu', pre_MLP=True, MLP_num=3)
    pass
    aa = ExtraMlp(256,256,256,2)
    x = torch.rand(256,256)
    out = aa(x)
    print(out - x)
