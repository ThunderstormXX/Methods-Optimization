"""
Стабильный N-Conjugate Frank–Wolfe (NFW)
с опциональным line search и нормализацией направлений.
"""

import torch
from torch.optim import Optimizer
from scipy.optimize import minimize_scalar


class NConjugateFrankWolfe(Optimizer):
    def __init__(
        self,
        params,
        constraint_set,
        n_conjugates=3,
        lr=0.2,
        use_real_hessian=False,
        use_line_search=False,
    ):
        self.constraint_set = constraint_set
        self.n_conjugates = n_conjugates
        self.use_real_hessian = use_real_hessian
        self.use_line_search = use_line_search

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.S_list = []
        self.D_list = []
        self.D_grad_list = []
        self.gamma_list = []
        self.iter = 0

    @torch.no_grad()
    def step(self, closure):
        """Один шаг NFW."""
        self.iter += 1

        # --- вычисляем loss и градиенты на текущих весах ---
        with torch.enable_grad():
            loss = closure()

        grads, params_dict = {}, {}
        for group in self.param_groups:
            if "params_dict" in group:
                for name, p in group["params_dict"].items():
                    if p.grad is not None:
                        grads[name] = p.grad.clone()
                        params_dict[name] = p

        # --- LMO ---
        s = self.constraint_set.lmo(grads)
        d = {name: s[name] - p for name, p in params_dict.items()}

        # --- ограничиваем по норме, чтобы не вылетать ---
        for name in d:
            norm = torch.norm(d[name])
            if norm > 1e-8:
                d[name] = d[name] / norm  # нормализация направления

        # --- обновляем историю направлений ---
        if len(self.S_list) >= self.n_conjugates:
            self.S_list.pop(0)
            self.D_list.pop(0)
        self.S_list.append(s)
        self.D_list.append(d)

        # --- комбинируем последние направления ---
        weights = torch.linspace(1.0, 0.5, steps=len(self.D_list))
        weights /= weights.sum()
        d_combined = {
            name: sum(w * self.D_list[i][name] for i, w in enumerate(weights))
            for name in params_dict
        }

        # --- нормализация комбинированного направления ---
        max_norm = max(torch.norm(v).item() for v in d_combined.values())
        if max_norm > 1e-8:
            for name in d_combined:
                d_combined[name] /= max_norm

        # --- выбор шага ---
        if self.use_line_search:
            loss0 = loss.item() if hasattr(loss, "item") else float(loss)

            def f_line_search(gamma):
                tmp_loss = 0.0
                # создаём копию параметров, чтобы не портить оригиналы
                with torch.enable_grad():
                    for name, p in params_dict.items():
                        p_temp = p + gamma * d_combined[name]
                        tmp_loss += torch.sum(grads[name] * (p_temp - p))  # приближение по первому порядку
                return loss0 + tmp_loss.item()

            try:
                res = minimize_scalar(f_line_search, bounds=(0.0, 0.2), method="bounded")
                gamma = float(res.x)
            except Exception:
                gamma = min(self.param_groups[0]["lr"], 0.1)
        else:
            gamma = min(self.param_groups[0]["lr"], 1.0 / (self.iter ** 0.5))
            gamma = min(gamma, 0.2)

        self.gamma_list.append(gamma)

        # --- обновляем параметры ---
        for name, p in params_dict.items():
            p.add_(gamma * d_combined[name])

        return loss.item() if hasattr(loss, "item") else loss
