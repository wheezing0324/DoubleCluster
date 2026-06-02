"""
数据工具模块 - 支持Non-IID数据划分
"""

import torch
import numpy as np
import random
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)


class NonIIDDataPartitioner:
    """Non-IID数据划分器，支持T1和T2两种Non-IID划分方式"""
    
    def __init__(self, dataset_name='mnist', num_clients=100, non_iid_type='T1', non_iid_param=0.04):
        """
        初始化数据划分器
        
        Args:
            dataset_name: 数据集名称 ('mnist' 或 'cifar10')
            num_clients: 客户端数量
            non_iid_type: Non-IID类型 ('T1' 或 'T2')
            non_iid_param: Non-IID参数
                - T1: epsilon (0-1之间，越小越Non-IID)
                - T2: L_num (每个客户端的标签数量)
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.non_iid_type = non_iid_type
        self.non_iid_param = non_iid_param
        
        # 数据集参数
        if dataset_name == 'mnist':
            self.num_classes = 10
            self.input_shape = (1, 28, 28)
        elif dataset_name == 'cifar10':
            self.num_classes = 10
            self.input_shape = (3, 32, 32)
        elif dataset_name == 'cifar100':
            self.input_shape = (3, 32, 32)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def load_dataset(self):
        """加载数据集"""
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(
                './data/mnist', train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                './data/mnist', train=False, download=True, transform=transform
            )
        elif self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                './data/cifar10', train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                './data/cifar10', train=False, download=True, transform=transform
            )
        elif self.dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                './data/cifar100', train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                './data/cifar100', train=False, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        logger.info(f"Loaded {self.dataset_name} dataset: train={len(train_dataset)}, test={len(test_dataset)}")
        return train_dataset, test_dataset
    
    def partition_data(self, train_dataset):
        """划分Non-IID数据"""
        if self.non_iid_type == 'T1':
            return self._partition_type1(train_dataset)
        elif self.non_iid_type == 'T2':
            return self._partition_type2(train_dataset)
        else:
            raise ValueError(f"Unknown Non-IID type: {self.non_iid_type}")
    
    def _partition_type1(self, dataset):
        epsilon = self.non_iid_param
        total = len(dataset)
        targets = np.array([dataset[i][1] for i in range(total)])
        indices = np.arange(total)
        
        # IID 部分
        iid_size = int(total * epsilon)
        iid_indices = np.random.choice(indices, iid_size, replace=False)
        # Non-IID 部分
        non_iid_indices = np.setdiff1d(indices, iid_indices)
        
        # 对 Non-IID 数据按标签排序
        sorted_by_label = non_iid_indices[np.argsort(targets[non_iid_indices])]
        
        # 平均分成 num_clients 块
        chunk_size = len(sorted_by_label) // self.num_clients
        client_data = [[] for _ in range(self.num_clients)]
        
        # 分配 IID 部分（均匀）
        for i, idx in enumerate(iid_indices):
            client_data[i % self.num_clients].append(idx)
        
        # 分配 Non-IID 部分（顺序分块）
        for i in range(self.num_clients):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_clients - 1 else len(sorted_by_label)
            client_data[i].extend(sorted_by_label[start:end])
        
        # 确保无空客户端（可选：如果某个客户端仍为空，从 IID 中补）
        # ... 

        return [Subset(dataset, data) for data in client_data]
    
    def _partition_type2(self, dataset):
        """
        T2类型Non-IID划分
        每个客户端只分配L_num个标签
        """
        L_num = int(self.non_iid_param)
        total_size = len(dataset)
        targets = [dataset[i][1] for i in range(total_size)]
        
        # 按标签分组
        label_to_indices = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(targets):
            label_to_indices[label].append(idx)
        
        # 随机打乱每个标签的数据
        for label in range(self.num_classes):
            random.shuffle(label_to_indices[label])
        
        client_data = [[] for _ in range(self.num_clients)]
        
        # 为每个客户端分配L_num个标签
        all_labels = list(range(self.num_classes))
        
        for client_id in range(self.num_clients):
            # 随机选择L_num个标签
            selected_labels = random.sample(all_labels, min(L_num, self.num_classes))
            
            for label in selected_labels:
                if label_to_indices[label]:
                    # 从该标签中取一部分数据
                    # 尽量均匀分配
                    num_samples_for_client = len(label_to_indices[label]) // (self.num_clients // len(selected_labels) + 1)
                    num_samples_for_client = max(1, num_samples_for_client)
                    
                    samples = label_to_indices[label][:num_samples_for_client]
                    client_data[client_id].extend(samples)
                    
                    # 移除已分配的数据
                    label_to_indices[label] = label_to_indices[label][num_samples_for_client:]
        
        # 处理未分配的数据（分配给数据最少的客户端）
        remaining_indices = []
        for label, indices_list in enumerate(label_to_indices):
            remaining_indices.extend(indices_list)
        
        if remaining_indices:
            # 找到数据最少的客户端
            data_sizes = [len(data) for data in client_data]
            for idx in remaining_indices:
                min_client = np.argmin(data_sizes)
                client_data[min_client].append(idx)
                data_sizes[min_client] += 1
        
        # 确保每个客户端都有数据
        for i in range(self.num_clients):
            if len(client_data[i]) == 0:
                # 从第一个有数据的标签中取一个样本
                for label in range(self.num_classes):
                    if label_to_indices[label]:
                        client_data[i].append(label_to_indices[label].pop())
                        break
        
        # 创建客户端数据集
        client_datasets = [Subset(dataset, data) for data in client_data]
        
        # 统计信息
        data_sizes = [len(ds) for ds in client_datasets]
        logger.info(f"T2 partitioning (L_num={L_num}): data sizes - min={min(data_sizes)}, "
                    f"max={max(data_sizes)}, mean={np.mean(data_sizes):.1f}")
        
        # 统计每个客户端的标签分布
        label_distributions = []
        for ds in client_datasets:
            labels = [dataset[i][1] for i in ds.indices]
            unique_labels = len(set(labels))
            label_distributions.append(unique_labels)
        
        logger.info(f"Label distribution per client: min={min(label_distributions)}, "
                    f"max={max(label_distributions)}, mean={np.mean(label_distributions):.1f}")
        
        return client_datasets
    
    def get_test_loader(self, test_dataset, batch_size=128):
        """获取测试数据加载器"""
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return test_loader


def get_data_stats(client_datasets):
    """获取客户端数据统计信息"""
    stats = {
        'data_sizes': [len(ds) for ds in client_datasets],
        'total_samples': sum(len(ds) for ds in client_datasets),
        'num_clients': len(client_datasets)
    }
    
    # 计算均值和标准差
    if stats['data_sizes']:
        stats['mean_size'] = np.mean(stats['data_sizes'])
        stats['std_size'] = np.std(stats['data_sizes'])
        stats['min_size'] = min(stats['data_sizes'])
        stats['max_size'] = max(stats['data_sizes'])
    
    return stats


def create_iid_partition(dataset, num_clients):
    """创建IID数据划分"""
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # 均匀划分
    client_data = []
    samples_per_client = total_size // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else total_size
        client_data.append(Subset(dataset, indices[start_idx:end_idx]))
    
    return client_data


def visualize_data_distribution(client_datasets, dataset, save_path=None):
    """可视化客户端数据分布"""
    import matplotlib.pyplot as plt
    
    num_clients = len(client_datasets)
    num_classes = 10
    
    # 获取每个客户端的标签分布
    label_distribution = np.zeros((num_clients, num_classes))
    
    for i, ds in enumerate(client_datasets):
        labels = [dataset[j][1] for j in ds.indices]
        for label in labels:
            label_distribution[i, label] += 1
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    plt.imshow(label_distribution, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Number of samples')
    plt.xlabel('Label Class')
    plt.ylabel('Client ID')
    plt.title('Data Distribution Across Clients')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    return label_distribution


# 简单的测试代码
if __name__ == '__main__':
    # 测试数据划分
    partitioner = NonIIDDataPartitioner(
        dataset_name='cifar100',
        num_clients=10,
        non_iid_type='T2',
        non_iid_param=2
    )
    
    train_dataset, test_dataset = partitioner.load_dataset()
    client_datasets = partitioner.partition_data(train_dataset)
    
    print(f"Created {len(client_datasets)} clients")
    for i, ds in enumerate(client_datasets[:5]):  # 只显示前5个
        print(f"Client {i}: {len(ds)} samples")
    
    # 获取测试加载器
    test_loader = partitioner.get_test_loader(test_dataset)
    print(f"Test loader has {len(test_loader)} batches")