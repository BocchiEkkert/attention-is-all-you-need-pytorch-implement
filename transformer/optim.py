'''定义优化器的学习率调度策略'''
import numpy as np

class ScheduledOptim():

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.d_model = d_model
        self.lr_mul = lr_mul

    def step(self):
        "更新参数和学习率"
        self._update_learning_rate()
        self._optimizer.step()
    
    def zero_grad(self):
        "梯度归零"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        "根据当前步数计算学习率缩放因子"
        return (self.d_model ** -0.5) * min(
            self.n_current_steps ** -0.5,
            self.n_current_steps * (self.n_warmup_steps ** -1.5)
        )
    
    def _update_learning_rate(self):
        "更新学习率"
        self.n_current_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr