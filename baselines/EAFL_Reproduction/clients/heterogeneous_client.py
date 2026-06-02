import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from collections import OrderedDict

class HeterogeneousClient:
    """异构客户端实现"""
    
    def __init__(self, client_id, model, dataset, device, speed_factor=1.0):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.device = device
        self.speed_factor = speed_factor
        self.staleness = 0
        self.last_participation_iter = 0
        self.local_epochs = 1
        self.batch_size = 64
        
        # 创建数据加载器
        if len(dataset) > 0:
            self.dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
        else:
            self.dataloader = None
        
        # 统计信息
        self.total_training_time = 0
        self.participation_count = 0
        
    def local_train(self, global_model_params, num_epochs=1, lr=0.005):
        """执行本地训练"""
        if len(self.dataset) == 0:
            # 空数据集，返回零梯度
            zero_gradients = []
            for param in self.model.parameters():
                zero_gradients.append(torch.zeros_like(param.data))
            return zero_gradients, 0
        
        # 模拟异构计算速度
        if self.speed_factor < 1.0:
            # 速度越慢，训练时间越长
            delay = (1.0 - self.speed_factor) * np.random.uniform(0.5, 1.5)
            #time.sleep(delay)
        
        start_time = time.time()
        
        # 加载全局模型
        self.model.load_state_dict(global_model_params)
        self.model.train()
        self.model.to(self.device)
        
        # 设置优化器
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练
        for epoch in range(num_epochs):
            running_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = running_loss / batch_count
        
        # 计算梯度（模型参数与初始参数的差值）
        local_gradients = self._compute_gradients(global_model_params)
        
        # 记录训练时间
        training_time = time.time() - start_time
        self.total_training_time += training_time
        self.participation_count += 1
        
        return local_gradients, len(self.dataset)
    
    def _compute_gradients(self, initial_params):
        """计算梯度（参数差）"""
        gradients = []
        for init_param, current_param in zip(initial_params.values(), self.model.parameters()):
            # 梯度 = 初始参数 - 当前参数
            grad = init_param.to(self.device) - current_param.data
            gradients.append(grad)
        return gradients
    
    def update_model(self, new_params):
        """更新本地模型"""
        self.model.load_state_dict(new_params)
        self.model.to(self.device)
        
    def get_gradients_with_global_model(self, global_model_params):
        """
        基于给定的全局模型进行一次本地训练，返回梯度（用于聚类）
        Args:
            global_model_params: 全局模型的 state_dict
        Returns:
            list of torch.Tensor: 每个参数的梯度
        """
        if len(self.dataset) == 0:
            # 空数据集返回零梯度
            return [torch.zeros_like(p) for p in self.model.parameters()]

        # 保存当前模型状态，以便恢复
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # 加载全局模型
        self.model.load_state_dict(global_model_params)
        self.model.train()
        self.model.to(self.device)
        
        # 使用一个 mini-batch 进行训练
        data, target = next(iter(self.dataloader))
        data, target = data.to(self.device), target.to(self.device)
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.005)  # 使用配置的学习率
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 收集梯度
        gradients = [p.grad.clone() for p in self.model.parameters()]
        
        # 清除梯度，恢复模型原始状态
        self.model.zero_grad()
        self.model.load_state_dict(original_state)
        
        return gradients

    def set_staleness(self, staleness):
        """设置staleness"""
        self.staleness = staleness
        
    def get_staleness(self):
        return self.staleness
    
    def get_data_size(self):
        return len(self.dataset)
    
    def get_average_training_time(self):
        """获取平均训练时间"""
        if self.participation_count > 0:
            return self.total_training_time / self.participation_count
        return 0.0
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_training_time = 0
        self.participation_count = 0