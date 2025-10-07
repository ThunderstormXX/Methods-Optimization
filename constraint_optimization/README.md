# Фреймворк ограниченной оптимизации для PyTorch

Модульный фреймворк для обучения нейронных сетей с ограничениями на параметры.

## 🎯 Особенности

- **Модульная архитектура**: независимые компоненты для множеств ограничений и оптимизаторов
- **Совместимость с PyTorch**: работает с любыми моделями `nn.Module`
- **Готовые оптимизаторы**: Frank-Wolfe и Projected Gradient
- **Расширяемость**: легко добавить новые типы ограничений и оптимизаторов

## 📁 Структура проекта

```
constraint_optimization/
├── main.py                # Точка входа, обучение и логирование
├── sets/
│   ├── __init__.py
│   └── l1_constraint.py   # Класс множества L1-нормы
├── optim/
│   ├── __init__.py
│   ├── frank_wolfe.py     # Класс оптимизатора Frank-Wolfe
│   └── projected_grad.py  # Класс оптимизатора Projected Gradient
└── models/
    ├── __init__.py
    └── small_cnn.py       # Небольшая CNN для MNIST
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск обучения

```bash
python main.py
```

Скрипт автоматически:
1. Загрузит датасет MNIST
2. Обучит две модели (с Frank-Wolfe и Projected Gradient)
3. Построит график сравнения
4. Сохранит результат в `optimization_comparison.png`

## 💡 Использование

### Пример с собственной моделью

```python
from constraint_optimization.models import SmallCNN
from constraint_optimization.sets import L1ConstraintSet
from constraint_optimization.optim import FrankWolfe

# Создаем модель
model = SmallCNN()

# Определяем ограничения: ||w||_1 <= tau для каждого параметра
tau = {name: 5.0 for name, _ in model.named_parameters()}
constraint = L1ConstraintSet(model, tau)

# Создаем оптимизатор
optimizer = FrankWolfe(
    [{"params": model.parameters(), 
      "params_dict": dict(model.named_parameters())}], 
    constraint, 
    lr=0.05
)

# Цикл обучения
for inputs, targets in dataloader:
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        return loss
    
    loss = optimizer.step(closure)
```

## 🔧 Компоненты

### Множества ограничений (sets)

#### `L1ConstraintSet`

Ограничение на L1-норму параметров: \(\|w\|_1 \leq \tau\)

**Методы:**
- `lmo(grad_dict)` — Linear Minimization Oracle для Frank-Wolfe
- `project(param_dict)` — проекция на множество для Projected Gradient

### Оптимизаторы (optim)

#### `FrankWolfe`

Метод условного градиента (Conditional Gradient):
\[
w_{t+1} = (1 - \alpha_t) w_t + \alpha_t s_t
\]
где \(s_t = \arg\min_{s \in \mathcal{C}} \langle \nabla f(w_t), s \rangle\)

**Параметры:**
- `lr` — скорость обучения (\(\alpha_t\))

#### `ProjectedGradient`

Проецированный градиентный спуск:
\[
w_{t+1} = \Pi_{\mathcal{C}}(w_t - \alpha_t \nabla f(w_t))
\]

**Параметры:**
- `lr` — скорость обучения

## 📊 Результаты

После запуска `main.py` вы увидите:

- Прогресс обучения для обоих методов
- График сравнения сходимости
- Финальные значения loss
- L1-нормы параметров (должны удовлетворять ограничению \(\leq \tau\))

## 🎨 Расширения

### Добавление нового типа ограничений

Создайте класс в `sets/` с методами:
- `lmo(grad_dict)` — для Frank-Wolfe
- `project(param_dict)` — для Projected Gradient

Пример для L2-шара:

```python
class L2ConstraintSet:
    def __init__(self, model, tau_per_param):
        self.tau = tau_per_param
    
    def lmo(self, grad_dict):
        # Для L2: s = -tau * grad / ||grad||_2
        s = {}
        for name, g in grad_dict.items():
            norm = torch.norm(g)
            s[name] = -self.tau[name] * g / (norm + 1e-8)
        return s
    
    def project(self, param_dict):
        # Проекция на L2-шар
        proj = {}
        for name, p in param_dict.items():
            norm = torch.norm(p)
            if norm <= self.tau[name]:
                proj[name] = p.clone()
            else:
                proj[name] = self.tau[name] * p / norm
        return proj
```

### Добавление нового оптимизатора

Наследуйтесь от `torch.optim.Optimizer` и реализуйте метод `step(closure)`.

## 📝 Литература

- **Frank-Wolfe**: Jaggi, M. (2013). "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization"
- **Projected Gradient**: Bertsekas, D. (1999). "Nonlinear Programming"

## 🤝 Вклад

Вы можете расширить фреймворк:
- Добавить новые типы ограничений (L∞, nuclear norm, etc.)
- Реализовать адаптивные стратегии выбора шага
- Добавить логирование в MLflow/Weights & Biases
- Реализовать стохастические версии алгоритмов

## 📄 Лицензия

MIT License

