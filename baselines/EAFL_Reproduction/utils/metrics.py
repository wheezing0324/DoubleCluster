import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_accuracy(model, test_loader, device):
    """计算准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    model.train()
    return correct / total if total > 0 else 0

def compute_loss(model, test_loader, device, criterion):
    """计算损失"""
    model.eval()
    total_loss = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            total += data.size(0)
    
    model.train()
    return total_loss / total if total > 0 else 0

def compute_precision_recall_f1(model, test_loader, device, num_classes=10):
    """计算精确率、召回率和F1分数"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    model.train()
    
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return precision, recall, f1

def compute_staleness_statistics(clients):
    """计算staleness统计信息"""
    staleness_values = [client.get_staleness() for client in clients]
    return {
        'mean': np.mean(staleness_values),
        'std': np.std(staleness_values),
        'max': np.max(staleness_values),
        'min': np.min(staleness_values)
    }

def compute_gradient_norm(gradients):
    """计算梯度范数"""
    total_norm = 0
    for grad in gradients:
        total_norm += torch.norm(grad).item() ** 2
    return np.sqrt(total_norm)