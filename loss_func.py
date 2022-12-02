import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class CELossWithLogits(nn.Module):
    """
    CE loss baseline
    """

    def __init__(self, class_counts: Union[list, np.array]):
        # def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super(CELossWithLogits, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = F.one_hot(targets, self.num_labels)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = torch.exp(logits)[:, None, :].sum(axis=-1) 

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()

class SeesawLossWithLogits(nn.Module):
    """
    unofficial implementation for Seesaw loss
    Args:
    class_counts: The list which has number of samples for each class. 
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super(SeesawLossWithLogits, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets, **kwargs):
        targets = F.one_hot(targets, self.num_labels)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class FederatedImbalanceLoss(nn.Module):
    """
    Args:
    class_counts: The list which has number of samples for each class. 
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """
    def __init__(
        self, 
        class_counts: Union[list, np.array], 
        p: float = 0.8,
        clamp_thres: float = 0,
        tao: float = 1
        ):
        super(FederatedImbalanceLoss, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.global_factor = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6
        self.clamp_thres = clamp_thres
        self.tao = tao
    
    def proto_factor(self, local_proto, global_proto):
        """
        [C, D]: D is 64 or 4
        """
        # factor = 1
        factor = torch.norm(global_proto.detach()-local_proto, p=1, dim=-1, keepdim=False) # [C]
        # factor = factor/torch.norm(factor, dim=-1, keepdim=False)
        print('factor \n min: {:f}\t mean: {:f}\t max: {:f}'.format(
            torch.min(factor), 
            torch.mean(factor), 
            torch.max(factor))
        )
        factor_mean = torch.mean(factor)
        factor_std = torch.std(factor)
        factor_refined = (factor - factor_mean + self.eps)/(factor_std+self.eps) + 1 - self.eps
        factor_refined = torch.clamp(factor_refined, min=self.clamp_thres) # 0 or 1
        print('factor_refined \n min: {:f}\t mean: {:f}\t max: {:f}'.format(
            torch.min(factor_refined), 
            torch.mean(factor_refined), 
            torch.max(factor_refined))
        )
        return factor_refined # [C]
    
    def proto_factor_cosine(self, local_proto, global_proto):
        """
        [C, D]: D is 64 or 4
        """
        # factor = 1
        norm_local = torch.norm(local_proto, dim=-1, keepdim=False)
        norm_global = torch.norm(global_proto.detach(), dim=-1, keepdim=False) # [C]
        factor_refined = torch.sum(local_proto*global_proto.detach(), dim=-1, keepdim=False)/(norm_local*norm_global+self.eps)
        
        # print('factor_refined \n min: {:f}\t mean: {:f}\t max: {:f}'.format(
        #     torch.min(factor_refined), 
        #     torch.mean(factor_refined), 
        #     torch.max(factor_refined)
        #     )
        # )
        # print(factor_refined)
        return factor_refined # [C]

    # def sample_factor(self, sample_proto, global_proto):
    #     """
    #     [C, D]: D is 512 or 4
    #     """
    #     # factor = 1
    #     factor = torch.norm(global_proto.detach()-sample_proto, dim=-1, keepdim=False) #
    #     factor_mean = torch.mean(factor)
    #     factor_std = torch.std(factor)
    #     factor_refined = (factor - factor_mean)/factor_std + 1
    #     factor_refined = torch.clamp(factor_refined, min=0)
    #     # factor_refined = torch.clamp(factor_refined, min=1)
    #     return factor_refined # [C]
    
    def forward(self, logits, targets, local_proto, global_proto):
        targets = F.one_hot(targets, self.num_labels) # [N, C]
        self.global_factor = self.global_factor.to(targets.device) # [C, C]
        max_element, _ = logits.max(axis=-1)
        # [N, C]
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits) # [N, C]
        denominator = (
            (1 - targets)[:, None, :]
            * self.global_factor[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits) # [N, C]

        sigma = numerator / (denominator + self.eps) # [N, C]
        # proto factor
        # proto_factor = self.proto_factor(local_proto=local_proto, global_proto=global_proto)
        cosine_score = self.proto_factor_cosine(local_proto=local_proto, global_proto=global_proto)
        proto_factor = (1+self.tao)/(cosine_score+self.tao) #
        # print(proto_factor)
        # sum in categories
        loss = (- proto_factor.view(1, -1) * targets * torch.log(sigma + self.eps)).sum(-1) # [N]
        return loss.mean() # scalar

#########################

class FederatedImbalanceLoss_v2(nn.Module):
    """
    Args:
    class_counts: The list which has number of samples for each class. 
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """
    def __init__(
        self, 
        class_counts: Union[list, np.array], 
        p: float = 0.8,
        clamp_thres: float = 0,
        tao: float = 1
        ):
        super(FederatedImbalanceLoss_v2, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.global_factor = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6
        self.clamp_thres = clamp_thres
        self.tao = tao
    
    def proto_factor_cosine(self, source_proto, target_proto):
        """
        [C, D]: D is 64 or 4
        """
        # factor = 1
        norm_source = torch.norm(source_proto, dim=-1, keepdim=False)
        norm_target = torch.norm(target_proto.detach(), dim=-1, keepdim=False) # [C]
        factor_refined = torch.sum(source_proto*target_proto.detach(), dim=-1, keepdim=False)/(norm_source*norm_target+self.eps)
        return factor_refined # [C]
    
    def forward(self, logits, targets, local_proto, global_proto, batch_protos):
        targets = F.one_hot(targets, self.num_labels) # [N, C]
        self.global_factor = self.global_factor.to(targets.device) # [C, C]
        max_element, _ = logits.max(axis=-1)
        # [N, C]
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits) # [N, C]
        denominator = (
            (1 - targets)[:, None, :]
            * self.global_factor[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits) # [N, C]

        sigma = numerator / (denominator + self.eps) # [N, C]
        # proto factor
        # proto_factor = self.proto_factor(local_proto=local_proto, global_proto=global_proto)
        cosine_score1 = self.proto_factor_cosine(source_proto=local_proto, target_proto=global_proto)
        cosine_score2_list = []
        for idx in range(logits.shape[0]):
            cosine_score_tmp = self.proto_factor_cosine(source_proto=batch_protos[idx], target_proto=global_proto)
            cosine_score2_list.append(cosine_score_tmp)
        #
        cosine_score2 = torch.mean(
            torch.stack(cosine_score2_list, dim=0),
            dim=0,
            keepdim=False
        )
        cosine_score = (cosine_score1 + cosine_score2)/2.0

        proto_factor = (1+self.tao)/(cosine_score+self.tao) #
        # print(proto_factor)
        # sum in categories
        loss = (- proto_factor.view(1, -1) * targets * torch.log(sigma + self.eps)).sum(-1) # [N]
        return loss.mean() # scalar

#########################


def global_avg_proto(local_protos):
    # local_protos: client_num*C*D
    return torch.mean(local_protos, dim=0, keepdim=False) # C*D

def global_gaussian_proto(local_protos):
    # local_protos: client_num*C*D
    mean = torch.mean(local_protos, dim=0, keepdim=False)
    std = torch.clamp(
        torch.std(local_protos, dim=0, keepdim=False),
        min=1
        )
    sample = torch.randn(mean.shape).to(mean.device)
    return sample * std + mean # C*D

###########################
def proto_factor_cosine(local_proto, global_proto):
    """
    [C, D]: D is 64 or 4
    """
    # factor = 1
    norm_local = torch.norm(local_proto, dim=-1, keepdim=False)
    norm_global = torch.norm(global_proto, dim=-1, keepdim=False) # [C]
    factor_refined = torch.sum(local_proto*global_proto, dim=-1, keepdim=False)/(norm_local*norm_global+1e-6)
    return factor_refined # [C]

def tao_func(cosine_score, tao):
    proto_factor = (1+tao)/(cosine_score+tao) #
    return proto_factor