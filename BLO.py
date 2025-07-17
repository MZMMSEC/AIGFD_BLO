import copy
import torch
from torch.cuda import amp

from loss import L2R_Loss, Fidelity_Loss_multi, categorical_focal_loss_fidelity



class BL_Optimizer:
    def __init__(self, model, device, train_tasks, pri_tasks, weight_init_EXIF=1., weight_init_face=1., num_exif_scale=3):
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.meta_weights = torch.tensor([weight_init_EXIF, weight_init_EXIF, weight_init_EXIF, weight_init_EXIF,
                                          weight_init_EXIF,weight_init_EXIF, weight_init_EXIF, weight_init_EXIF, weight_init_EXIF,
                                          weight_init_face, weight_init_face],
                                         requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks

        self.criterion = L2R_Loss(num_scale=num_exif_scale)
        self.criterion_ce = categorical_focal_loss_fidelity()
        self.criterion_bcelogits = Fidelity_Loss_multi()

    def virtual_step(self, train_x, train_y, joint_texts, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        logits_all_image, logits_all_text = self.model(train_x, joint_texts)

        _, train_loss = self.model_fit(logits_all_image, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())#, allow_unused=True)

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, train_x, train_y, joint_texts,
                          val_x, val_y,
                          alpha, alpha_lambda, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, joint_texts, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]


        with amp.autocast(enabled=False):
            logits_all_image, logits_all_text = self.model_(val_x, joint_texts)

            _, val_loss = self.model_fit(logits_all_image, val_y)

            loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())

        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)

        hessian = self.compute_hessian(d_model, train_x, train_y, joint_texts)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha_lambda * h

    def compute_hessian(self, d_model, train_x, train_y, joint_texts):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d


        logits_all_image, logits_all_text = self.model(train_x, joint_texts)

        _, train_loss = self.model_fit(logits_all_image, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        logits_all_image, logits_all_text = self.model(train_x, joint_texts)
        _, train_loss = self.model_fit(logits_all_image, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, logits_all_image, targets):
        ## for EXIF-ordinal tasks
        logits_all_EXIF_ordinal = logits_all_image[:, :12]
        iso_logits = logits_all_EXIF_ordinal[:, 0:3]
        av_logits = logits_all_EXIF_ordinal[:, 3:6]
        et_logits = logits_all_EXIF_ordinal[:, 6:9]
        fl_logits = logits_all_EXIF_ordinal[:, 9:12]
        ## EXIF: learning to rank
        loss_iso = self.criterion(iso_logits, targets['iso'].cuda())
        loss_av = self.criterion(av_logits, targets['av'].cuda())
        loss_et = self.criterion(et_logits, targets['et'].cuda())
        loss_fl = self.criterion(fl_logits, targets['fl'].cuda())
        loss_exif_ordinal = loss_iso + loss_av + loss_et + loss_fl

        ## for EXIF-categorical tasks
        logits_makes = logits_all_image[:, 12: 22]
        logits_mm = logits_all_image[:, 22: 30]
        logits_em = logits_all_image[:, 30: 37]
        logits_wb = logits_all_image[:, 37: 39]
        logits_ep = logits_all_image[:, 39: 45]
        ### multiclass classification
        loss_makes = self.criterion_ce(logits_makes, targets['makes'].cuda(), num_classes=logits_makes.shape[1])
        loss_mm = self.criterion_ce(logits_mm, targets['mm'].cuda(), num_classes=logits_mm.shape[1])
        loss_em = self.criterion_ce(logits_em, targets['em'].cuda(), num_classes=logits_em.shape[1])
        loss_wb = self.criterion_ce(logits_wb, targets['wb'].cuda(), num_classes=logits_wb.shape[1])
        loss_ep = self.criterion_ce(logits_ep, targets['ep'].cuda(), num_classes=logits_ep.shape[1])
        loss_exif_categorical = loss_makes + loss_mm + loss_em + loss_wb + loss_ep

        loss_exif = loss_exif_ordinal + loss_exif_categorical

        # for face task: multilabel classification
        logits_all_face2text = logits_all_image[:, 45:]
        loss_face2text_coarse = self.criterion_bcelogits(logits_all_face2text[:, :2], targets['face2text'][:, 1].cuda(),
                                                         num_classes=2)
        loss_face2text_fine = self.criterion_bcelogits(logits_all_face2text[:, 2:],
                                                       targets['face2text'][:, 2:].float().cuda(), num_classes=None)
        loss_face2text = loss_face2text_coarse + loss_face2text_fine
        loss_face = loss_face2text

        total_loss = loss_exif + loss_face # this will be not used though it is returned
        all_loss = [loss_iso, loss_av, loss_et, loss_fl,
                    loss_makes, loss_mm, loss_em, loss_wb, loss_ep,
                    loss_face2text_coarse, loss_face2text_fine]

        return total_loss, all_loss