"""
Множество ограничений на L∞-норму параметров модели.
"""

import torch
from .base import ConstraintSet


class LinfConstraintSet(ConstraintSet):
    """
    Множество: ||w||_∞ <= tau
    
    Гиперкуб [-tau, tau]^d.
    """
    
    def lmo(self, grad_dict):
        """
        LMO для L∞-шара (гиперкуба).
        
        Решение: s_i = -tau * sign(grad_i) для всех i
        (все координаты на границе)
        
        Parameters
        ----------
        grad_dict : dict[str, torch.Tensor]
            Словарь градиентов
            
        Returns
        -------
        dict[str, torch.Tensor]
            Вершина гиперкуба
        """
        s = {}
        for name, g in grad_dict.items():
            # Каждая координата принимает значение ±tau в зависимости от знака градиента
            s[name] = -self.tau[name] * torch.sign(g)
        return s
    
    def project(self, param_dict):
        """
        Проекция на L∞-шар (покоординатный clamp).
        
        proj(w)_i = clip(w_i, -tau, tau)
        
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
            proj[name] = torch.clamp(p, -self.tau[name], self.tau[name])
        return proj
    
    def check_feasibility(self, param_dict, tolerance=1e-5):
        """
        Проверяет, что ||w||_∞ <= tau + tolerance.
        
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
            max_abs = torch.max(torch.abs(p)).item()
            results[name] = max_abs <= self.tau[name] + tolerance
        return results

