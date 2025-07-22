import numpy as np
import torch


class OnlineMeanCovBatch:
    """
    在线计算批量数据的均值和协方差矩阵
    """
    
    def __init__(self, dim):
        """
        初始化
        
        Args:
            dim (int): 数据维度
        """
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.cov = np.eye(dim)
        self.M2 = np.zeros((dim, dim))  # 用于计算协方差的辅助矩阵
        
    def update(self, data):
        """
        更新统计信息
        
        Args:
            data (np.ndarray): 新的数据批次，形状为 (batch_size, dim)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        batch_size = data.shape[0]
        
        for i in range(batch_size):
            self.count += 1
            sample = data[i]
            
            # 更新均值
            delta = sample - self.mean
            self.mean += delta / self.count
            
            # 更新协方差相关的统计量
            delta2 = sample - self.mean
            self.M2 += np.outer(delta, delta2)
            
            # 更新协方差矩阵
            if self.count > 1:
                self.cov = self.M2 / (self.count - 1)
    
    def get_mean(self):
        """
        获取当前均值
        
        Returns:
            np.ndarray: 均值向量
        """
        return self.mean.copy()
    
    def get_cov(self):
        """
        获取当前协方差矩阵
        
        Returns:
            np.ndarray: 协方差矩阵
        """
        return self.cov.copy()
    
    def save(self, filename):
        """
        保存统计信息到文件
        
        Args:
            filename (str): 保存文件名
        """
        np.savez(filename, 
                mean=self.mean, 
                cov=self.cov, 
                count=self.count,
                M2=self.M2)
    
    def load(self, filename):
        """
        从文件加载统计信息
        
        Args:
            filename (str): 文件名
        """
        try:
            data = np.load(filename)
            self.mean = data['mean']
            self.cov = data['cov']
            self.count = data['count'].item()
            if 'M2' in data:
                self.M2 = data['M2']
            else:
                # 如果没有M2，从协方差重新计算
                self.M2 = self.cov * (self.count - 1) if self.count > 1 else np.zeros((self.dim, self.dim))
                
        except FileNotFoundError:
            print(f"Warning: Could not load statistics from {filename}. Using default values.")
            # 使用默认值
            self.mean = np.zeros(self.dim)
            self.cov = np.eye(self.dim)
            self.count = 0
            self.M2 = np.zeros((self.dim, self.dim))
        except Exception as e:
            print(f"Error loading statistics from {filename}: {e}")
            print("Using default values.")
            self.mean = np.zeros(self.dim)
            self.cov = np.eye(self.dim)
            self.count = 0
            self.M2 = np.zeros((self.dim, self.dim))
    
    def reset(self):
        """
        重置所有统计信息
        """
        self.count = 0
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        self.M2 = np.zeros((self.dim, self.dim))


class OnlineMeanStdBatch:
    """
    在线计算批量数据的均值和标准差
    """
    
    def __init__(self, dim):
        """
        初始化
        
        Args:
            dim (int): 数据维度
        """
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)  # 方差
        self.M2 = np.zeros(dim)  # 用于计算方差的辅助变量
        
    def update(self, data):
        """
        更新统计信息
        
        Args:
            data (np.ndarray): 新的数据批次，形状为 (batch_size, dim)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        batch_size = data.shape[0]
        
        for i in range(batch_size):
            self.count += 1
            sample = data[i]
            
            # 使用Welford算法更新均值和方差
            delta = sample - self.mean
            self.mean += delta / self.count
            delta2 = sample - self.mean
            self.M2 += delta * delta2
            
            # 更新方差
            if self.count > 1:
                self.var = self.M2 / (self.count - 1)
    
    def get_mean(self):
        """
        获取当前均值
        
        Returns:
            np.ndarray: 均值向量
        """
        return self.mean.copy()
    
    def get_std(self):
        """
        获取当前标准差
        
        Returns:
            np.ndarray: 标准差向量
        """
        return np.sqrt(self.var)
    
    def get_var(self):
        """
        获取当前方差
        
        Returns:
            np.ndarray: 方差向量
        """
        return self.var.copy()
    
    def save(self, filename):
        """
        保存统计信息到文件
        
        Args:
            filename (str): 保存文件名
        """
        np.savez(filename, 
                mean=self.mean, 
                var=self.var, 
                count=self.count,
                M2=self.M2)
    
    def load(self, filename):
        """
        从文件加载统计信息
        
        Args:
            filename (str): 文件名
        """
        try:
            data = np.load(filename)
            self.mean = data['mean']
            self.var = data['var']
            self.count = data['count'].item()
            if 'M2' in data:
                self.M2 = data['M2']
            else:
                self.M2 = self.var * (self.count - 1) if self.count > 1 else np.zeros(self.dim)
                
        except FileNotFoundError:
            print(f"Warning: Could not load statistics from {filename}. Using default values.")
            # 使用默认值
            self.mean = np.zeros(self.dim)
            self.var = np.ones(self.dim)
            self.count = 0
            self.M2 = np.zeros(self.dim)
        except Exception as e:
            print(f"Error loading statistics from {filename}: {e}")
            print("Using default values.")
            self.mean = np.zeros(self.dim)
            self.var = np.ones(self.dim)
            self.count = 0
            self.M2 = np.zeros(self.dim)
    
    def reset(self):
        """
        重置所有统计信息
        """
        self.count = 0
        self.mean = np.zeros(self.dim)
        self.var = np.ones(self.dim)
        self.M2 = np.zeros(self.dim) 