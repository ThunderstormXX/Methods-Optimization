"""
Множество ограничений на L2-норму параметров модели.
"""

import torch
from .base import ConstraintSet


class L2ConstraintSet(ConstraintSet):
    """
    Множество: ||w||_2 <= tau
    
    Евклидов шар радиуса tau.
    """
    
    def lmo(self, grad_dict):
        """
        LMO для L2-шара.
        
        Решение: s = -tau * grad / ||grad||_2
        
        Parameters
        ----------
        grad_dict : dict[str, torch.Tensor]
            Словарь градиентов
            
        Returns
        -------
        dict[str, torch.Tensor]
            Точка минимума на границе L2-шара
        """
        s = {}
        for name, g in grad_dict.items():
            norm = torch.norm(g)
            if norm > 1e-10:
                s[name] = -self.tau[name] * g / norm
            else:
                # Если градиент нулевой, возвращаем нулевой вектор
                s[name] = torch.zeros_like(g)
        return s
    
    def project(self, param_dict):
        """
        Проекция на L2-шар.
        
        Если ||w||_2 <= tau: proj(w) = w
        Если ||w||_2 > tau: proj(w) = tau * w / ||w||_2
        
        Parameters
        ----------
        param_dict : dict[str, torch.Tensor]
            Словарь параметров
            
        Returns
        -------
        dict[str, torch.Tensor]
            Спроецированные параметры
        """
        proj = {}
        for name, p in param_dict.items():
            norm = torch.norm(p)
            
            if norm <= self.tau[name]:
                proj[name] = p.clone()
            else:
                proj[name] = self.tau[name] * p / norm
        
        return proj
    
    def check_feasibility(self, param_dict, tolerance=1e-5):
        """
        Проверяет, что ||w||_2 <= tau + tolerance.
        
        Parameters
        ----------
        param_dict : dict[str, torch.Tensor]
            Словарь параметров
        tolerance : float
            Допустимая погрешность
            
        Returns
        -------
        dict[str, bool]
            Результаты проверки для каждого параметра
        """
        results = {}
        for name, p in param_dict.items():
            norm = torch.norm(p).item()
            results[name] = norm <= self.tau[name] + tolerance
        return results

