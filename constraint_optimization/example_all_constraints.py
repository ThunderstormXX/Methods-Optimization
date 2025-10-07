"""
Мини-эксперимент: сравнение L1, L2 и L∞ ограничений
с оптимизаторами Frank–Wolfe, Projected Gradient и N-Conjugate FW.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.small_cnn import SmallCNN
from sets.l1_constraint import L1ConstraintSet
from sets.l2_constraint import L2ConstraintSet
from sets.linf_constraint import LinfConstraintSet
from optim.frank_wolfe import FrankWolfe
from optim.projected_grad import ProjectedGradient
from optim.n_conjugate_frank_wolfe import NConjugateFrankWolfe  # 🔥 новый оптимизатор


# ======================= Утилиты =======================
def train(model, opt, data, loss_fn, device, epochs=10):
    """Обучает модель с tqdm-индикатором."""
    model.train()
    losses = []
    for epoch in range(epochs):
        pbar = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            def closure():
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                return loss

            loss = opt.step(closure)
            losses.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})
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


def evaluate_constraint_strength(model, tau_dict):
    """Проверяет, не слишком ли маленькие или большие радиусы множеств."""
    norms_ = norms(model)
    for name, n in norms_.items():
        t = tau_dict[name]
        ratio_l1 = n["l1"] / t
        ratio_l2 = n["l2"] / t
        if ratio_l1 < 0.1 and ratio_l2 < 0.1:
            print(f"⚠️ {name}: радиус слишком большой (ограничение неактивно)")
        elif ratio_l1 > 1.5 or ratio_l2 > 1.5:
            print(f"⚠️ {name}: радиус слишком мал (ограничение зажимает веса)")
    return norms_


# ======================= Основной код =======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Данные ---
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(2000)), batch_size=64, shuffle=True
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Конфигурации ---
    constraints = {
        "L1": (L1ConstraintSet, 1.0),
        "L2": (L2ConstraintSet, 5.0),
        "L∞": (LinfConstraintSet, 0.1),
    }
    opts = {
        "FW": FrankWolfe,
        "PG": ProjectedGradient,
        "NFW": lambda params, C: NConjugateFrankWolfe(
            params, C,
            n_conjugates=3,
            lr=0.1,
            use_real_hessian=False,
            use_line_search=True,  # можно и False
        ),
    }

    results = {}
    for cname, (Cset, tau_val) in constraints.items():
        for oname, Opt in opts.items():
            key = f"{cname}+{oname}"
            print(f"\n=== {key} ===")
            model = SmallCNN().to(device)
            tau = {n: tau_val for n, _ in model.named_parameters()}
            C = Cset(model, tau)

            # поддержка lambda для NFW
            if callable(Opt):
                opt = Opt([{"params": model.parameters(), "params_dict": dict(model.named_parameters())}], C)
            else:
                opt = Opt(
                    [{"params": model.parameters(), "params_dict": dict(model.named_parameters())}],
                    C, lr=0.05
                )

            losses = train(model, opt, loader, loss_fn, device, epochs=20)
            results[key] = dict(
                losses=losses,
                norms=evaluate_constraint_strength(model, tau),
                constraint=C,
                model=model,
                tau=tau,
            )
            print(f"Final loss = {losses[-1]:.3f}")

        # --- 📊 Отдельные графики для каждого множества ---
    for cname in constraints.keys():
        plt.figure(figsize=(7, 5))
        for oname in opts.keys():
            key = f"{cname}+{oname}"
            if key in results:
                plt.plot(results[key]["losses"], label=oname)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{cname}-Constraint: Comparison of Optimizers")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"comparison_{cname.replace('∞','inf')}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ График для {cname} сохранён как {filename}")

    # --- 🧩 Дополнительно: общий график для всех ---
    plt.figure(figsize=(7, 5))
    for k, r in results.items():
        plt.plot(r["losses"], label=k)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("All Constraints and Optimizers")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_all.png", dpi=150)
    print("\n📉 Общий график сохранён как comparison_all.png")



if __name__ == "__main__":
    main()
