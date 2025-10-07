import torch
from .base import ConstraintSet


class L1ConstraintSet(ConstraintSet):
    """Множество: ||w||_1 <= tau"""

    def lmo(self, grad_dict):
        s = {}
        for name, g in grad_dict.items():
            g_flat = g.flatten()
            idx = torch.argmax(torch.abs(g_flat))
            s_vec = torch.zeros_like(g_flat)
            s_vec[idx] = -self.tau[name] * torch.sign(g_flat[idx])
            s[name] = s_vec.view_as(g)
        return s

    def project(self, param_dict):
        proj = {}
        for name, p in param_dict.items():
            flat = p.flatten()
            norm1 = flat.abs().sum()
            if norm1 <= self.tau[name]:
                proj[name] = p.clone()
            else:
                u = flat.abs()
                sorted_u, _ = torch.sort(u, descending=True)
                cssv = torch.cumsum(sorted_u, dim=0)
                indices = torch.arange(1, len(sorted_u) + 1, device=p.device)
                rho_mask = sorted_u * indices > (cssv - self.tau[name])
                if rho_mask.any():
                    rho = torch.nonzero(rho_mask, as_tuple=False)[-1].item()
                    theta = (cssv[rho] - self.tau[name]) / (rho + 1.0)
                else:
                    theta = 0.0
                w = torch.sign(flat) * torch.clamp(u - theta, min=0)
                proj[name] = w.view_as(p)
        return proj

    def check_feasibility(self, param_dict, tolerance=1e-5):
        results = {}
        for name, p in param_dict.items():
            norm1 = p.abs().sum().item()
            results[name] = norm1 <= self.tau[name] + tolerance
        return results
