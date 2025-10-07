"""
Оптимизатор Projected Gradient Descent для ограниченной оптимизации.
"""

import torch


class ProjectedGradient(torch.optim.Optimizer):
    """
    Оптимизатор Projected Gradient Descent.
    
    Делает градиентный шаг, затем проецирует результат на допустимое множество.
    """
    
    def __init__(self, params, constraint_set, lr=1e-3):
        """
        Инициализация оптимизатора Projected Gradient.
        
        Parameters
        ----------
        params : iterable
            Параметры модели для оптимизации
        constraint_set : ConstraintSet
            Множество ограничений с методом project()
        lr : float
            Скорость обучения (размер шага градиента)
        """
        self.constraint_set = constraint_set
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure):
        """
        Выполняет один шаг оптимизации Projected Gradient.
        
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
        
        # Собираем градиенты и делаем градиентный шаг
        new_params = {}
        params_dict = {}
        lr = self.param_groups[0]['lr']
        
        for group in self.param_groups:
            if 'params_dict' in group:
                for name, param in group['params_dict'].items():
                    if param.grad is not None:
                        params_dict[name] = param
                        new_params[name] = param - lr * param.grad
        
        # Проецируем на множество ограничений
        proj = self.constraint_set.project(new_params)
        
        # Обновляем параметры
        for name, param in params_dict.items():
            param.copy_(proj[name])
        
        return loss.item() if hasattr(loss, 'item') else loss

