import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(output, target, alpha, gamma, normalize_alpha=True):
    # output: [batch_size, num_classes, ...] -> [batch_size, num_classes]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1))  # 直接展平空间维度
    else:
        output = output.view(output.size(0), -1)  # 确保是 [batch_size, num_classes]
    
    # 将alpha转换为tensor并确保与output在同一设备上
    alpha = torch.tensor(alpha, dtype=torch.float32, device=output.device)
    
    # 对alpha进行归一化（可选）
    if normalize_alpha:
        alpha = alpha / alpha.sum()  # 归一化，使alpha之和为1
    
    ce_loss = F.cross_entropy(output, target, reduction='none')  # [batch_size]
    target_one_hot = F.one_hot(target, num_classes=output.size(1)).float()  # [batch_size, num_classes]
    probs = output.softmax(dim=1)  # [batch_size, num_classes]
    pt = (probs * target_one_hot).sum(dim=1)  # [batch_size]
    
    # 根据target的类别索引选择对应的alpha值
    alpha_t = alpha[target]  # [batch_size]
    focal_weight = alpha_t * (1 - pt) ** gamma  # [batch_size]
    return (focal_weight * ce_loss).mean()


def mse_loss(output, target):
    # output: [batch_size, 1, ...] -> [batch_size, 1]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1))  # 展平空间维度
    else:
        output = output.view(output.size(0), -1)  # 确保是 [batch_size, 1]
    
    output = torch.sigmoid(output)  # [batch_size, 1]
    return nn.MSELoss()(output, target)
