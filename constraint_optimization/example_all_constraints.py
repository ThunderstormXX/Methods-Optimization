"""
Мини-эксперимент: сравнение L1, L2 и L∞ ограничений
с оптимизаторами Frank–Wolfe и Projected Gradient.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from models.small_cnn import SmallCNN
from sets.l1_constraint import L1ConstraintSet
from sets.l2_constraint import L2ConstraintSet
from sets.linf_constraint import LinfConstraintSet
from optim.frank_wolfe import FrankWolfe
from optim.projected_grad import ProjectedGradient


# ======================= Утилиты =======================
def train(model, opt, data, loss_fn, device, epochs=3):
    model.train()
    losses = []
    for _ in range(epochs):
        for x, y in data:
            x, y = x.to(device), y.to(device)

            def closure():
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                return loss

            losses.append(opt.step(closure))   # 🔧 без .item()
    return losses

def norms(model):
    """Возвращает L1, L2, L∞ нормы параметров."""
    return {
        name: dict(
            l1=param.abs().sum().item(),
            l2=param.norm().item(),
            linf=param.abs().max().item(),
        )
        for name, param in model.named_parameters()
    }


def check(feas, tau, typ):
    for name, ok in feas.items():
        print(f"{typ} {name}: {'✓' if ok else '✗'}")


# ======================= Основной код =======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Данные ---
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(1000)), 64, True)
    loss_fn = nn.CrossEntropyLoss()

    # --- Конфигурации ---
    constraints = {
        "L1": (L1ConstraintSet, 10.0),
        "L2": (L2ConstraintSet, 5.0),
        "L∞": (LinfConstraintSet, 0.1),
    }
    opts = {
        "FW": FrankWolfe,
        "PG": ProjectedGradient,
    }

    results = {}
    for cname, (Cset, tau_val) in constraints.items():
        for oname, Opt in opts.items():
            key = f"{cname}+{oname}"
            print(f"\n=== {key} ===")
            model = SmallCNN().to(device)
            tau = {n: tau_val for n, _ in model.named_parameters()}
            C = Cset(model, tau)
            opt = Opt([{"params": model.parameters(), "params_dict": dict(model.named_parameters())}], C, lr=0.05)
            losses = train(model, opt, loader, loss_fn, device)
            results[key] = dict(losses=losses, norms=norms(model), constraint=C, model=model, tau=tau)
            print(f"Final loss = {losses[-1]:.3f}")

    # --- Графики ---
    plt.figure(figsize=(7, 5))
    for k, r in results.items():
        plt.plot(r["losses"], label=k)
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("comparison.png", dpi=150)
    print("\nГрафик сохранен как comparison.png")

    # --- Проверка ограничений ---
    print("\n=== Проверка ограничений ===")
    for k, r in results.items():
        feas = r["constraint"].check_feasibility(dict(r["model"].named_parameters())) \
            if hasattr(r["constraint"], "check_feasibility") else {}
        print(f"\n{k}:")
        check(feas, r["tau"], k.split("+")[0])


if __name__ == "__main__":
    main()
