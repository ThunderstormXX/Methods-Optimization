"""
Оптимизатор Frank-Wolfe (Conditional Gradient) для ограниченной оптимизации.
"""

import torch


class FrankWolfe(torch.optim.Optimizer):
    """
    Оптимизатор Frank-Wolfe для выпуклых множеств.
    
    Использует Linear Minimization Oracle (LMO) для нахождения направления движения
    и комбинирует текущую точку с вершиной множества.
    """
    
    def __init__(self, params, constraint_set, lr=1e-3):
        """
        Инициализация оптимизатора Frank-Wolfe.
        
        Parameters
        ----------
        params : iterable
            Параметры модели для оптимизации
        constraint_set : ConstraintSet
            Множество ограничений с методом lmo()
        lr : float
            Скорость обучения (шаг alpha_t)
        """
        self.constraint_set = constraint_set
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure):
        """
        Выполняет один шаг оптимизации Frank-Wolfe.
        
        Parameters
        ----------
        closure : callable
            Функция, которая вычисляет loss и градиенты
            
        Returns
        -------
        float
            Значение функции потерь
        """
        # Вычисляем loss и градиенты
        with torch.enable_grad():
            loss = closure()
        
        # Собираем градиенты
        grads = {}
        params_dict = {}
        for group in self.param_groups:
            if 'params_dict' in group:
                for name, param in group['params_dict'].items():
                    if param.grad is not None:
                        grads[name] = param.grad.clone()
                        params_dict[name] = param
        
        # LMO: находим вершину s множества
        s = self.constraint_set.lmo(grads)
        
        # Update rule: w_{t+1} = (1 - lr) * w_t + lr * s
        lr = self.param_groups[0]['lr']
        for name, param in params_dict.items():
            param.copy_((1 - lr) * param + lr * s[name])
        
        return loss.item() if hasattr(loss, 'item') else loss

