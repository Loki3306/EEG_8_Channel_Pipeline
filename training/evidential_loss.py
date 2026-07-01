import torch
import torch.nn as nn
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device):
    """
    Computes the KL divergence between a Dirichlet distribution with parameters alpha
    and a uniform Dirichlet distribution.
    """
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    Evidential Deep Learning MSE Loss.
    Args:
        output: Model predictions (Evidence), shape (B, K)
        target: True labels (one-hot), shape (B, K)
        epoch_num: Current training epoch
        num_classes: Number of classes (K)
        annealing_step: Step at which KL divergence weight reaches 1.0
    """
    if device is None:
        device = output.device
        
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    p = alpha / S
    
    err = (target - p) ** 2
    var = p * (1 - p) / (S + 1)
    
    loss_mse = torch.sum(err + var, dim=1, keepdim=True)
    
    # KL Divergence regularization (penalize evidence on incorrect classes)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    # Calculate modified alpha for KL (only penalize non-target classes)
    alpha_tilde = target + (1 - target) * alpha
    
    loss_kl = kl_divergence(alpha_tilde, num_classes, device=device)
    
    loss = loss_mse + annealing_coef * loss_kl
    return loss.mean()

class EvidentialLoss(nn.Module):
    def __init__(self, num_classes=2, annealing_step=10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        
    def forward(self, output, target, epoch_num):
        """
        output: Raw network outputs [B, K]
        target: Class indices [B] or one-hot [B, K]
        """
        device = output.device
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).float()
            
        return edl_mse_loss(output, target, epoch_num, self.num_classes, self.annealing_step, device)
