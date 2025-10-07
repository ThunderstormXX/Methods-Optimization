"""
Основной скрипт для обучения моделей с ограниченной оптимизацией.

Сравнивает Frank-Wolfe и Projected Gradient на задаче MNIST.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.small_cnn import SmallCNN
from sets.l1_constraint import L1ConstraintSet
from optim.frank_wolfe import FrankWolfe
from optim.projected_grad import ProjectedGradient


def main():
    """Основная функция для запуска обучения и визуализации."""
    
    # === Загрузка данных MNIST ===
    print("Загрузка данных MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Используем подмножество для быстрого обучения
    subset = torch.utils.data.Subset(trainset, range(1000))
    trainloader = torch.utils.data.DataLoader(
        subset, 
        batch_size=64, 
        shuffle=True
    )
    
    # === Определение устройства ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    
    # === Функция потерь ===
    criterion = torch.nn.CrossEntropyLoss()
    
    # === Функция обучения ===
    def train(model, optimizer, n_epochs=5):
        """
        Обучает модель с заданным оптимизатором.
        
        Parameters
        ----------
        model : nn.Module
            Модель для обучения
        optimizer : torch.optim.Optimizer
            Оптимизатор
        n_epochs : int
            Количество эпох
            
        Returns
        -------
        list
            История значений функции потерь
        """
        model.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    return loss
                
                loss = optimizer.step(closure)
                losses.append(loss)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            print(f"  Эпоха {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    # === Обучение с Frank-Wolfe ===
    print("\n" + "="*60)
    print("Обучение с Frank-Wolfe")
    print("="*60)
    
    model_fw = SmallCNN().to(device)
    tau_fw = {name: 5.0 for name, _ in model_fw.named_parameters()}
    constraint_fw = L1ConstraintSet(model_fw, tau_fw)
    
    opt_fw = FrankWolfe(
        [{"params": model_fw.parameters(), 
          "params_dict": dict(model_fw.named_parameters())}], 
        constraint_fw, 
        lr=0.05
    )
    
    fw_losses = train(model_fw, opt_fw, n_epochs=5)
    
    # === Обучение с Projected Gradient ===
    print("\n" + "="*60)
    print("Обучение с Projected Gradient")
    print("="*60)
    
    model_pg = SmallCNN().to(device)
    tau_pg = {name: 5.0 for name, _ in model_pg.named_parameters()}
    constraint_pg = L1ConstraintSet(model_pg, tau_pg)
    
    opt_pg = ProjectedGradient(
        [{"params": model_pg.parameters(), 
          "params_dict": dict(model_pg.named_parameters())}], 
        constraint_pg, 
        lr=0.05
    )
    
    pg_losses = train(model_pg, opt_pg, n_epochs=5)
    
    # === Визуализация ===
    print("\nПостроение графика сравнения...")
    plt.figure(figsize=(12, 6))
    
    plt.plot(fw_losses, label="Frank-Wolfe", alpha=0.7, linewidth=1.5)
    plt.plot(pg_losses, label="Projected Gradient", alpha=0.7, linewidth=1.5)
    
    plt.xlabel("Итерация", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Сравнение методов ограниченной оптимизации на MNIST", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig("optimization_comparison.png", dpi=150)
    print("График сохранен как 'optimization_comparison.png'")
    
    plt.show()
    
    # === Итоговая статистика ===
    print("\n" + "="*60)
    print("Итоговая статистика:")
    print("="*60)
    print(f"Frank-Wolfe - финальный loss: {fw_losses[-1]:.4f}")
    print(f"Projected Gradient - финальный loss: {pg_losses[-1]:.4f}")
    
    # Проверка L1-норм параметров
    print("\nL1-нормы параметров (должны быть <= tau = 5.0):")
    print("\nFrank-Wolfe:")
    for name, param in model_fw.named_parameters():
        l1_norm = param.abs().sum().item()
        print(f"  {name}: {l1_norm:.4f}")
    
    print("\nProjected Gradient:")
    for name, param in model_pg.named_parameters():
        l1_norm = param.abs().sum().item()
        print(f"  {name}: {l1_norm:.4f}")


if __name__ == "__main__":
    main()

