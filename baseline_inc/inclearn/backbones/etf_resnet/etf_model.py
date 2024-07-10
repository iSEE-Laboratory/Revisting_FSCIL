from inclearn.backbones.etf_resnet.ETFHead import ETFHead
from inclearn.backbones.etf_resnet.mlp_ffn_neck import MLPFFNNeck
# from resnet18 import ResNet18
from inclearn.backbones.etf_resnet.resnet12 import ResNet12
import torch.nn as nn
import torch


class etf_model(nn.Module):
    def __init__(self, state_dict_path):
        super(etf_model, self).__init__()
        self.backbone = ResNet12()
        self.neck = MLPFFNNeck(640, 512)
        self.head = ETFHead(100, 512)
        state_dict = torch.load(state_dict_path, 'cpu')['state_dict']

        err = self.load_state_dict(state_dict, strict=False)
        print(err)

    def forward(self, x):
        feat = self.backbone(x)
        feat_2 = self.neck(feat)
        logits = self.head(feat_2)
        return {'logits': logits}

    def forward_feature(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    model = etf_model().eval()
    print(model)
    a = torch.ones((1, 3, 84, 84))
    print(a.shape)
    logits = model(a)
    print(logits)
    print(logits.shape)
    pass
