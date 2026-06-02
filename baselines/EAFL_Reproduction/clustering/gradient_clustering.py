import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging

class GradientClustering:
    """基于梯度相似度的动态聚类"""
    
    def __init__(self, num_clusters=5, similarity_threshold=0.7):
        self.num_clusters = num_clusters
        self.similarity_threshold = similarity_threshold
        self.clusters = None
        self.cluster_heads = None
        self.re_clustering_interval = 100
        self.logger = logging.getLogger(__name__)
        
    def flatten_gradients(self, gradients_list):
        """展平梯度列表"""
        flat_grads = []
        for grads in gradients_list:
            # 将所有梯度展平并连接
            flat = torch.cat([g.flatten() for g in grads]).cpu().numpy()
            flat_grads.append(flat)
        return np.array(flat_grads)
    
    def compute_gradient_similarity(self, gradients_list):
        """计算梯度间的余弦相似度"""
        flat_grads = self.flatten_gradients(gradients_list)
        
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(flat_grads)
        
        return similarity_matrix
    
    def cluster_clients(self, gradients_list):
        """基于梯度相似度进行K-Means聚类"""
        if len(gradients_list) < self.num_clusters:
            self.logger.warning(f"Number of clients ({len(gradients_list)}) less than clusters ({self.num_clusters})")
            # 每个客户端单独成簇
            clusters = {i: [i] for i in range(len(gradients_list))}
            cluster_heads = {i: i for i in range(len(gradients_list))}
            self.clusters = clusters
            self.cluster_heads = cluster_heads
            return clusters, cluster_heads
        
        # 展平梯度
        flat_grads = self.flatten_gradients(gradients_list)
        
        # 处理NaN或Inf值
        flat_grads = np.nan_to_num(flat_grads, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # K-Means聚类
            kmeans = KMeans(
                n_clusters=self.num_clusters, 
                random_state=42, 
                n_init=10,
                max_iter=300
            )
            cluster_labels = kmeans.fit_predict(flat_grads)
            
            # 构建聚类结果
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            
            # 为每个聚类选择聚类头（选择最接近中心的客户端）
            cluster_heads = {}
            for label, clients in clusters.items():
                # 找到离聚类中心最近的客户端
                center = kmeans.cluster_centers_[label]
                min_dist = float('inf')
                head = clients[0]
                
                for client_idx in clients:
                    client_grad = flat_grads[client_idx]
                    dist = np.linalg.norm(client_grad - center)
                    if dist < min_dist:
                        min_dist = dist
                        head = client_idx
                
                cluster_heads[label] = head
            
            self.clusters = clusters
            self.cluster_heads = cluster_heads
            
            self.logger.info(f"Clustering completed: {len(clusters)} clusters created")
            for label, clients in clusters.items():
                self.logger.debug(f"Cluster {label}: {len(clients)} clients")
            
            return clusters, cluster_heads
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            # 回退到简单分组
            clusters = {}
            for i in range(len(gradients_list)):
                cluster_id = i % self.num_clusters
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(i)
            cluster_heads = {cid: clients[0] for cid, clients in clusters.items()}
            self.clusters = clusters
            self.cluster_heads = cluster_heads
            return clusters, cluster_heads
    
    def update_clustering(self, gradients_list, iteration, re_clustering_interval):
        """动态更新聚类"""
        if iteration % re_clustering_interval == 0 or iteration == 1:
            self.logger.info(f"Updating clustering at iteration {iteration}")
            return self.cluster_clients(gradients_list)
        return self.clusters, self.cluster_heads
    
    def compute_intra_cluster_similarity(self, gradients_list):
        """计算聚类内平均相似度"""
        if self.clusters is None:
            return 0.0
        
        similarities = []
        flat_grads = self.flatten_gradients(gradients_list)
        
        for cluster_id, clients in self.clusters.items():
            if len(clients) > 1:
                cluster_grads = flat_grads[clients]
                # 计算聚类内平均余弦相似度
                sim_matrix = cosine_similarity(cluster_grads)
                # 取上三角平均值（不包括对角线）
                mask = np.triu_indices_from(sim_matrix, k=1)
                if len(mask[0]) > 0:
                    avg_sim = np.mean(sim_matrix[mask])
                    similarities.append(avg_sim)
        
        return np.mean(similarities) if similarities else 0.0