import torch
import psutil
import GPUtil
import numpy as np
from typing import Dict, Any

class GPUMemoryProfiler:
    """GPU显存占用分析工具"""
    
    def __init__(self):
        self.baseline_memory = 0
        self.memory_log = []
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取详细的GPU显存信息"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        # PyTorch显存信息
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        memory_cached = torch.cuda.memory_cached() / (1024**3)       # GB
        
        # 系统级GPU信息
        try:
            gpus = GPUtil.getGPUs()
            gpu = gpus[0] if gpus else None
            total_memory = gpu.memoryTotal / 1024 if gpu else 0  # GB
            used_memory = gpu.memoryUsed / 1024 if gpu else 0    # GB
            free_memory = gpu.memoryFree / 1024 if gpu else 0    # GB
        except:
            total_memory = used_memory = free_memory = 0
            
        return {
            "torch_allocated": memory_allocated,
            "torch_reserved": memory_reserved, 
            "torch_cached": memory_cached,
            "system_total": total_memory,
            "system_used": used_memory,
            "system_free": free_memory,
            "utilization": (used_memory / total_memory * 100) if total_memory > 0 else 0
        }
    
    def profile_batch_size_impact(self, model, input_shape, batch_sizes=[1, 64, 256, 1024, 4096]):
        """分析不同batch size对显存的影响"""
        results = {}
        
        # 清空缓存
        torch.cuda.empty_cache()
        baseline = self.get_gpu_memory_info()["torch_allocated"]
        
        for batch_size in batch_sizes:
            try:
                torch.cuda.empty_cache()
                
                # 创建测试输入
                test_input = torch.randn(batch_size, *input_shape, device='cuda')
                
                # 记录推理前显存
                before_memory = self.get_gpu_memory_info()["torch_allocated"]
                
                # 执行推理
                with torch.no_grad():
                    _ = model(test_input)
                
                # 记录推理后显存
                after_memory = self.get_gpu_memory_info()["torch_allocated"]
                
                results[batch_size] = {
                    "input_memory": test_input.element_size() * test_input.numel() / (1024**3),
                    "total_memory": after_memory,
                    "inference_memory": after_memory - before_memory,
                    "memory_per_sample": (after_memory - before_memory) / batch_size
                }
                
                # 清理
                del test_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                results[batch_size] = {"error": str(e)}
                torch.cuda.empty_cache()
                
        return results
    
    def analyze_ppo_memory_usage(self, num_envs=4096, obs_dim=241, action_dim=23, hidden_dims=[1024, 1024, 512]):
        """分析PPO训练的理论显存占用"""
        
        # 1. 模型参数显存 (固定)
        actor_params = obs_dim * hidden_dims[0]  # 输入层
        for i in range(len(hidden_dims)-1):
            actor_params += hidden_dims[i] * hidden_dims[i+1]  # 隐藏层
        actor_params += hidden_dims[-1] * action_dim  # 输出层
        
        critic_params = actor_params  # 假设critic结构相似
        total_params = (actor_params + critic_params) * 4 / (1024**3)  # FP32, GB
        
        # 2. 激活值显存 (与batch size相关)
        activations_per_sample = sum(hidden_dims) + action_dim  # 估算
        activations_memory = num_envs * activations_per_sample * 4 / (1024**3)  # FP32, GB
        
        # 3. PPO特有缓存
        ppo_buffers = num_envs * (obs_dim + action_dim * 3 + 5) * 4 / (1024**3)  # 各种状态缓存
        
        # 4. PyTorch开销 (经验值: 20-30%)
        overhead = (total_params + activations_memory + ppo_buffers) * 0.25
        
        total_estimated = total_params + activations_memory + ppo_buffers + overhead
        
        return {
            "model_parameters": total_params,
            "activations": activations_memory,
            "ppo_buffers": ppo_buffers,
            "pytorch_overhead": overhead,
            "total_estimated": total_estimated,
            "breakdown": {
                "parameters": f"{total_params/total_estimated*100:.1f}%",
                "activations": f"{activations_memory/total_estimated*100:.1f}%",
                "buffers": f"{ppo_buffers/total_estimated*100:.1f}%",
                "overhead": f"{overhead/total_estimated*100:.1f}%"
            }
        }
    
    def print_memory_report(self, analysis):
        """打印详细的显存分析报告"""
        print("=" * 60)
        print("GPU MEMORY ANALYSIS REPORT")
        print("=" * 60)
        print(f"模型参数:     {analysis['model_parameters']:.2f} GB ({analysis['breakdown']['parameters']})")
        print(f"激活值:       {analysis['activations']:.2f} GB ({analysis['breakdown']['activations']})")
        print(f"PPO缓存:      {analysis['ppo_buffers']:.2f} GB ({analysis['breakdown']['buffers']})")
        print(f"PyTorch开销:  {analysis['pytorch_overhead']:.2f} GB ({analysis['breakdown']['overhead']})")
        print("-" * 60)
        print(f"预估总显存:   {analysis['total_estimated']:.2f} GB")
        print("=" * 60)

# 使用示例
def main():
    profiler = GPUMemoryProfiler()
    
    # 1. 当前显存状态
    current_memory = profiler.get_gpu_memory_info()
    print("当前GPU显存状态:")
    for key, value in current_memory.items():
        print(f"  {key}: {value:.2f} GB" if isinstance(value, float) else f"  {key}: {value}")
    
    # 2. PPO显存分析
    ppo_analysis = profiler.analyze_ppo_memory_usage(
        num_envs=4096,  # 您的环境数量
        obs_dim=241,    # 观测维度
        action_dim=23,  # 动作维度
    )
    profiler.print_memory_report(ppo_analysis)
    
    # 3. Batch size影响分析
    print("\nBatch Size影响分析:")
    print("Batch Size | 激活值显存 | 总显存占用")
    print("-" * 40)
    for batch_size in [1024, 2048, 4096, 8192]:
        analysis = profiler.analyze_ppo_memory_usage(num_envs=batch_size)
        print(f"{batch_size:9d} | {analysis['activations']:8.2f} GB | {analysis['total_estimated']:8.2f} GB")

if __name__ == "__main__":
    main() 