import torch
from your_model_def import ActorModel  # 替换为你自己的模型定义

# 加载 checkpoint
checkpoint = torch.load("your_model.pth", map_location="cpu")

# 恢复模型
model = ActorModel()
model.load_state_dict(checkpoint['actor_model_state_dict'])
model.eval()

# 转换成 TorchScript
scripted_model = torch.jit.script(model)

# 保存为 TorchScript 模型
scripted_model.save("policy.pt")
