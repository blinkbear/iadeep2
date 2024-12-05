import torch
import torch.nn as nn
from transformers import ResNetForImageClassification
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# 初始化 PyNVML
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # 根据实际情况设置 GPU 索引

# 定义 hook 函数
start_times = {}
end_times = {}
mem_before = {}
mem_after = {}

def hook_fn_forward(module, input, output):
    start_times[module] = time.time()
    mem_before[module] = torch.cuda.memory_allocated()
    # 获取显存和 GPU 利用率
    s=nvmlDeviceGetUtilizationRates(handle).__str__().split("(")[-1].split(")")[0]
    pairs = s.split(', ')
    data = {}
    for pair in pairs:
        key, value = pair.split(': ')
        value = int(value.strip('%'))
        data[key] = value
    module.memory_util_start = data['memory'] 
    module.gpu_util_start = data['gpu'] 
    
def hook_fn_backward(module, grad_input, grad_output):
    end_times[module] = time.time()
    mem_after[module] = torch.cuda.memory_allocated()
    # 获取显存和 GPU 利用率
    s=nvmlDeviceGetUtilizationRates(handle).__str__().split("(")[-1].split(")")[0]
    pairs = s.split(', ')
    data = {}
    for pair in pairs:
        key, value = pair.split(': ')
        value = int(value.strip('%'))
        data[key] = value
    module.memory_util_end = data['memory'] 
    module.gpu_util_end = data['gpu'] 

# 加载模型
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
model = model.cuda()

# 为每一层注册 hook
for name, module in model.named_modules():
    if len(list(module.children())) == 0:
        module.register_forward_hook(hook_fn_forward)
        module.register_backward_hook(hook_fn_backward)

# 准备输入数据
input_tensor = torch.randn(1, 3, 224, 224).cuda()
labels = torch.tensor([0]).cuda()  # 假设类别标签为 0

# 前向传播
output = model(input_tensor, labels=labels)

# 计算损失
loss = output.loss

# 反向传播
loss.backward()

# 收集和打印每层的执行时间和显存占用情况
for name, module in model.named_modules():
    if len(list(module.children())) == 0:
        if hasattr(module, 'memory_util_start'):
            exec_time = end_times[module] - start_times[module]
            mem_used = mem_after[module] - mem_before[module]
            print(f"Layer: {name}")
            print(f"  Execution Time: {exec_time:.6f} seconds")
            print(f"  Memory Used: {mem_used / (1024**2):.2f} MB")
            print(f"  Memory Utilization Start: {module.memory_util_start}%")
            print(f"  Memory Utilization End: {module.memory_util_end}%")
            print(f"  GPU Utilization Start: {module.gpu_util_start}%")
            print(f"  GPU Utilization End: {module.gpu_util_end}%")
            print()