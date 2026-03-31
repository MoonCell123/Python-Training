import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    class TwoLayerNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.model(x)
    
    model = TwoLayerNet(784, 256, 1)
    x = torch.randn(32, 784)
    # 定义损失函数（二分类交叉熵）
    criterion = nn.BCELoss()
    # 定义优化器（随机梯度下降）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 真实标签（0或1）
    y_true = torch.randint(0, 2, (32, 1)).float()
    # ===== 反向传播完整流程 =====
    # 1. 清零梯度（防止梯度累加
    optimizer.zero_grad()
    # 2. 正向传播
    y_pred = model(x)
    # 3. 计算损失
    loss = criterion(y_pred, y_true)
    # 4. 反向传播（自动计算梯度）
    loss.backward()
    # 5. 更新参数
    optimizer.step()
    print(f"损失值: {loss.item():.4f}")
    print(f"预测值:\n{y_pred}")
    



