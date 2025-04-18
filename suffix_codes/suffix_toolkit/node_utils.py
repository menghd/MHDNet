import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, confusion_matrix
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train 函数
def train(model, dataloaders, optimizer, criterion_cls, criterion_reg, lambda_weights, cls_features, reg_features, task_node_map, out_nodes, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    cls_losses = {feature: 0.0 for feature in cls_features}
    reg_losses = {feature: 0.0 for feature in reg_features}
    
    data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))
    
    for _ in range(num_batches):
        optimizer.zero_grad()
        
        inputs_list = []
        labels_batch = None
        for node in sorted(data_iterators.keys()):
            inputs, labels = next(data_iterators[node])
            inputs_list.append(inputs.to(device))
            if labels_batch is None:
                labels_batch = labels
        
        outputs = model(inputs_list)
        
        total_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)
        
        for i, feature in enumerate(cls_features):
            node_idx = out_nodes.index(task_node_map[feature])
            cls_output = outputs[node_idx]
            cls_target = labels_batch[feature].to(device).long()
            loss_cls = criterion_cls(cls_output, cls_target, feature)
            cls_loss += lambda_weights['cls'][i] * loss_cls
            cls_losses[feature] += loss_cls.item()
        
        for i, feature in enumerate(reg_features):
            node_idx = out_nodes.index(task_node_map[feature])
            reg_output = outputs[node_idx]
            reg_target = labels_batch[feature].to(device).unsqueeze(1)
            loss_reg = criterion_reg(reg_output, reg_target)
            reg_loss += lambda_weights['reg'][i] * loss_reg
            reg_losses[feature] += loss_reg.item()
        
        total_loss = cls_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_reg_loss += reg_loss.item()
    
    avg_loss = running_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Total Loss: {avg_loss:.4f}")
    print("Classification Losses:")
    for feature in cls_features:
        print(f"  {feature}: {cls_losses[feature] / num_batches:.4f}")
    print("Regression Losses:")
    for feature in reg_features:
        print(f"  {feature}: {reg_losses[feature] / num_batches:.4f}")
    
    return running_loss / num_batches, running_cls_loss / num_batches, running_reg_loss / num_batches

# 修改后的 validate 函数
def validate(model, dataloaders, criterion_cls, criterion_reg, lambda_weights, cls_features, reg_features, task_node_map, out_nodes, epoch, num_epochs, focal_params):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    cls_preds = {feature: [] for feature in cls_features}
    cls_targets = {feature: [] for feature in cls_features}
    reg_preds = {feature: [] for feature in reg_features}
    reg_targets = {feature: [] for feature in reg_features}
    cls_losses = {feature: [] for feature in cls_features}
    reg_losses = {feature: [] for feature in reg_features}
    
    with torch.no_grad():
        data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
        num_batches = len(next(iter(data_iterators.values())))
        
        for _ in range(num_batches):
            inputs_list = []
            labels_batch = None
            for node in sorted(data_iterators.keys()):
                inputs, labels = next(data_iterators[node])
                inputs_list.append(inputs.to(device))
                if labels_batch is None:
                    labels_batch = labels
            
            outputs = model(inputs_list)
            
            total_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)
            reg_loss = torch.tensor(0.0, device=device)
            
            for feature in cls_features:
                node_idx = out_nodes.index(task_node_map[feature])
                cls_output = outputs[node_idx]
                cls_target = labels_batch[feature].to(device).long()
                loss_cls = criterion_cls(cls_output, cls_target, feature)
                cls_loss += lambda_weights['cls'][cls_features.index(feature)] * loss_cls
                
                if cls_output.dim() > 2:
                    cls_output = cls_output.view(cls_output.size(0), cls_output.size(1))
                cls_pred = torch.argmax(cls_output, dim=1)
                cls_preds[feature].append(cls_pred)
                cls_targets[feature].append(cls_target)
                cls_losses[feature].append(loss_cls.item())
            
            for feature in reg_features:
                node_idx = out_nodes.index(task_node_map[feature])
                reg_output = outputs[node_idx]
                reg_target = labels_batch[feature].to(device).unsqueeze(1)
                loss_reg = criterion_reg(reg_output, reg_target)
                reg_loss += lambda_weights['reg'][reg_features.index(feature)] * loss_reg
                
                if reg_output.dim() > 2:
                    reg_output = reg_output.view(reg_output.size(0), reg_output.size(1))
                reg_preds[feature].append(F.sigmoid(reg_output))
                reg_targets[feature].append(reg_target)
                reg_losses[feature].append(loss_reg.item())
            
            total_loss = cls_loss + reg_loss
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()
    
    # 合并预测和目标
    for feature in cls_features:
        cls_preds[feature] = torch.cat(cls_preds[feature]).cpu().numpy()
        cls_targets[feature] = torch.cat(cls_targets[feature]).cpu().numpy()
    
    for feature in reg_features:
        reg_preds[feature] = torch.cat(reg_preds[feature]).cpu().numpy()
        reg_targets[feature] = torch.cat(reg_targets[feature]).cpu().numpy()
    
    # 计算分类指标
    metrics = {'cls': {}, 'reg': {}}
    for feature in cls_features:
        num_classes = len(focal_params[feature][0])  # 从 focal_params 获取类别数
        cm = confusion_matrix(cls_targets[feature], cls_preds[feature], labels=list(range(num_classes)))
        
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        specificity = TN / (TN + FP + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        per_class_metrics = [
            {
                'Class': f'Class {i}',
                'Recall': recall[i],
                'Precision': precision[i],
                'Specificity': specificity[i],
                'F1': f1[i]
            }
            for i in range(num_classes)
        ]
        macro_metrics = {
            'Class': 'Avg',
            'Recall': np.mean(recall),
            'Precision': np.mean(precision),
            'Specificity': np.mean(specificity),
            'F1': np.mean(f1)
        }
        per_class_metrics.append(macro_metrics)
        
        metrics['cls'][feature] = {
            'per_class': per_class_metrics,
            'focal_loss': np.mean(cls_losses[feature])
        }
    
    # 计算回归指标
    for feature in reg_features:
        mse = mean_squared_error(reg_targets[feature], reg_preds[feature])
        metrics['reg'][feature] = {
            'mse': mse,
            'loss': np.mean(reg_losses[feature])
        }
    
    # 打印损失
    avg_loss = running_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Total Loss: {avg_loss:.4f}")
    print("Classification Losses:")
    for feature in cls_features:
        print(f"  {feature}: {np.mean(cls_losses[feature]):.4f}")
    print("Regression Losses:")
    for feature in reg_features:
        print(f"  {feature}: {np.mean(reg_losses[feature]):.4f}")
    
    # 打印指标
    print("\nValidation Metrics by Task and Feature:")
    for feature in cls_features:
        print(f"\nClassification - {feature}")
        headers = ["Class", "Recall", "Precision", "Specificity", "F1"]
        table = [
            [m['Class'], f"{m['Recall']:.4f}", f"{m['Precision']:.4f}", 
             f"{m['Specificity']:.4f}", f"{m['F1']:.4f}"]
            for m in metrics['cls'][feature]['per_class']
        ]
        print(tabulate(table, headers=headers, tablefmt="grid"))
    
    for feature in reg_features:
        print(f"\nRegression - {feature}")
        headers = ["Metric", "Value"]
        table = [["MSE", f"{metrics['reg'][feature]['mse']:.4f}"]]
        print(tabulate(table, headers=headers, tablefmt="grid"))
    
    return running_loss / num_batches, running_cls_loss / num_batches, running_reg_loss / num_batches, metrics
