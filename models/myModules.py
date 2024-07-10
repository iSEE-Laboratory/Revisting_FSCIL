import torch
from torch import nn
from timm.models.layers.helpers import to_2tuple

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
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeatureRectify(nn.Module):
    def __init__(self, feature_dim, act_layer=nn.GELU, drop=0.):
        super(FeatureRectify, self).__init__()
        self.mlp_inter = Mlp(feature_dim, hidden_features=512, out_features=feature_dim, act_layer=act_layer, drop=drop)
        self.mlp_final = Mlp(feature_dim, hidden_features=512, out_features=feature_dim, act_layer=act_layer, drop=drop)
        self.mlp_mixed = Mlp(feature_dim * 2, out_features=feature_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_inter, x_final):
        x_inter = self.mlp_inter(x_inter)
        x_final = self.mlp_final(x_final)
        x = torch.cat((x_inter, x_final), dim=1)
        res = self.mlp_mixed(x)
        return res
