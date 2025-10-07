"""
Базовые классы для множеств ограничений.
"""

from abc import ABC, abstractmethod
import torch


class ConstraintSet(ABC):
    """
    Абстрактный базовый класс для множеств ограничений.
    
    Все множества должны реализовать два метода:
    - lmo(): Linear Minimization Oracle для Frank-Wolfe
    - project(): Проекция на множество для Projected Gradient
    """
    
    def __init__(self, model, tau_per_param):
        """
        Инициализация множества ограничений.
        
        Parameters
        ----------
        model : torch.nn.Module
            Модель PyTorch
        tau_per_param : dict[str, float]
            Параметры ограничений для каждого параметра модели
        """
        self.tau = tau_per_param
        self.param_names = [name for name, _ in model.named_parameters()]
    
    @abstractmethod
    def lmo(self, grad_dict):
        """
        Linear Minimization Oracle.
        
        Решает задачу: argmin_{s ∈ C} ⟨grad, s⟩
        
        Parameters
        ----------
        grad_dict : dict[str, torch.Tensor]
            Словарь градиентов
            
        Returns
        -------
        dict[str, torch.Tensor]
            Точка минимума (вершина множества)
        """
        pass
    
    @abstractmethod
    def project(self, param_dict):
        """
        Проекция на множество ограничений.
        
        Решает задачу: argmin_{w ∈ C} ||w - param||²
        
        Parameters
        ----------
        param_dict : dict[str, torch.Tensor]
            Словарь параметров для проекции
            
        Returns
        -------
        dict[str, torch.Tensor]
            Спроецированные параметры
        """
        pass
    
    def check_feasibility(self, param_dict, tolerance=1e-5):
        """
        Проверяет, принадлежат ли параметры множеству.
        
        Parameters
        ----------
        param_dict : dict[str, torch.Tensor]
            Словарь параметров для проверки
        tolerance : float
            Допустимая погрешность
            
        Returns
        -------
        dict[str, bool]
            Словарь результатов проверки для каждого параметра
        """
        raise NotImplementedError("Метод check_feasibility не реализован")

