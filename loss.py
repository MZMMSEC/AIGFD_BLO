import torch
import torch.nn.functional as F


EPS = 1e-2
esp = 1e-8


class Fidelity_Loss_multi(torch.nn.Module):
    def __init__(self):
        super(Fidelity_Loss_multi, self).__init__()

    def forward(self, p, g, num_classes):
        if num_classes is None: # multilabel classification
            p = torch.sigmoid(p)
            loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))
            loss = torch.mean(loss, dim=1)

        else: # multiclass classification, using the softmax formulation
            p = F.softmax(p, dim=1)
            # 将g转化成one-hot编码
            g = g.view(-1)
            g_onehot = torch.zeros(g.size(0), num_classes).to(p.device)
            g_onehot.scatter_(1, g.unsqueeze(1), 1)  # 将类别对应的位置设为 1

            loss = torch.sqrt(p * g_onehot + esp)
            loss = 1 - loss.sum(1)

        return torch.mean(loss)



class categorical_focal_loss_fidelity(torch.nn.Module):
    'refer to https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py'
    'softmax formulation'
    def __init__(self):
        super(categorical_focal_loss_fidelity, self).__init__()
        self.gamma = 2.

    def forward(self, p, g, num_classes):
        p = F.softmax(p, dim=1)
        # 将g转化成one-hot编码
        g = g.view(-1)
        g_onehot = torch.zeros(g.size(0), num_classes).to(p.device)
        g_onehot.scatter_(1, g.unsqueeze(1), 1)  # 将类别对应的位置设为 1

        loss = torch.sqrt(p * g_onehot + esp)
        loss = 1 - loss.sum(1)
        p_max =  p[torch.arange(g.size(0)), g]
        f_loss = (1 - p_max) ** self.gamma * loss
        return torch.mean(f_loss)


class L2R_Loss(torch.nn.Module):
    '''
    y_pred: [bz]
    y: [bz] -- 需要输入的y是EXIF levels的gt number，不是one-hot labels
    '''

    def __init__(self, num_scale=3):
        super(L2R_Loss, self).__init__()
        self.num_scale = num_scale

    def prediction_expectation(self, y_pred):
        y_pred = F.softmax(y_pred, dim=1)
        # pdb.set_trace()
        if self.num_scale == 3:
            lvl_expt = y_pred[:, 0] * 1 + y_pred[:, 1] * 2 + y_pred[:, 2] * 3
        else:
            lvl_expt = y_pred[:, 0] * 1 + y_pred[:, 1] * 2 + y_pred[:, 2] * 3 + y_pred[:, 3] * 4 + y_pred[:, 4] * 5

        return lvl_expt

    def forward(self, y_pred, y):
        y_pred = self.prediction_expectation(y_pred)

        y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device) * 0.5
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

        return loss