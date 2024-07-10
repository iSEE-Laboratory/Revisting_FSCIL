import torch.nn as nn
import torch.nn.functional as F
import torch


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


def get_distance_matrix(x, type='distance', no_grad=False):
    if type == 'distance':
        d = pdist(x, squared=False)
        res = d
    else:
        assert type == 'angle'
        sd = (x.unsqueeze(0) - x.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        res = s_angle
    if no_grad:
        with torch.no_grad():
            res = res.detach()  # todo: necessary?
    return res


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            # mean_td = t_d[t_d > 0].mean()
            # t_d = t_d / mean_td

        d = pdist(student, squared=False)
        # mean_d = d[d > 0].mean()
        # d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


def pairwise_kl_div_v2(logit_x, logit_y):
    """
    :param logit_x:
    :param logit_y:
    :return: p(x)*[log(p(x)) - log(p(y))]
    """
    log_px = F.log_softmax(logit_x, dim=1)
    log_py = F.log_softmax(logit_y, dim=1)
    px = F.softmax(logit_x, dim=1)

    px = px.unsqueeze(1).transpose(1, 2)
    logit = log_px.unsqueeze(1) - log_py.unsqueeze(0)
    kl_div = logit.bmm(px)
    return kl_div
