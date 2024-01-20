"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.ones((batch_size * anchor_count, batch_size * anchor_count), device=device) - torch.eye(batch_size * anchor_count, device=device)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        somme = mask.sum(1)
        somme[somme==0]=1
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / somme

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class overparametrization_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, ratio_consistency = 0.9, ratio_balance = 0.1):
        super(overparametrization_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp
        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance
        
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32)).cuda()
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32)).cuda()
        self.init_param(mean=0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):

        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            ensembled_output = 0.5 * (output + output2).detach()
        else:
            output = outputs
            ensembled_output = output.detach()
        eps = 1e-4

        U_square = self.u[index]**2 * label 
        V_square = self.v[index]**2 * (1 - label) 
        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)
        E =  U_square - V_square
        self.E = E
        
        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(original_prediction + U_square - V_square.detach(), min = eps)
        prediction = F.normalize(prediction, p = 1, eps = eps)
        prediction = torch.clamp(prediction, min = eps, max = 1.0)
        label_one_hot = self.soft_to_hard(output.detach())
        MSE_loss = F.mse_loss((label_one_hot + U_square - V_square), label,  reduction='sum') / len(label)
        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim = -1))
        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = eps, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(index, output, output2)
            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return  loss.mean()

    def consistency_loss(self, index, output1, output2):            
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1,1), 1)
