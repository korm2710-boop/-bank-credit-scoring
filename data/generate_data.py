# Код для создания синтетических данных
import numpy as np
import pandas as pd
import os

# Фиксируем seed для воспроизводимости результатов
np.random.seed(42)
n_samples = 10000

# 1. Генерация базовых признаков клиентов
age = np.random.randint(21, 65, size=n_samples)
income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).astype(int) # Реалистичное логнормальное распределение доходов
credit_limit = (income * np.random.uniform(0.5, 3.0, size=n_samples)).astype(int)
utilization_rate = np.random.beta(a=2, b=5, size=n_samples) # Большая часть тратит умеренно, но есть пик к 100%

# Кредитная история: количество просрочек более 30 дней за последние 2 года
# Зависит от утилизации лимита и дохода (чем выше утилизация и ниже доход, тем больше просрочек)
delinquency_prob = utilization_rate * 0.4 + (1 / (income / 10000)) * 0.1
delinquencies = np.random.poisson(lam=delinquency_prob * 3, size=n_samples)
delinquencies = np.clip(delinquencies, 0, 10)

# 2. Моделирование целевой переменной (Default - невозврат кредита)
# Вероятность дефолта зависит от комбинации факторов (риск-логика)
logit_score = (
    -2.0 
    + 2.5 * utilization_rate 
    + 0.5 * delinquencies 
    - 0.03 * (age - 35) 
    - 0.5 * np.log1p(income / 10000)
)
prob_default = 1 / (1 + np.exp(-logit_score))
is_default = np.random.binomial(n=1, p=prob_default)

# 3. Финансовые показатели для бизнес-считалки
loan_amount = (credit_limit * np.random.uniform(0.6, 0.9, size=n_samples)).astype(int)
interest_rate = np.where(delinquencies > 1, 0.24, 0.16) # Риск-доходность: выше риск -> выше ставка

# Собираем датафрейм
df = pd.DataFrame({
    'client_id': range(1, n_samples + 1),
    'age': age,
    'monthly_income': income,
    'credit_limit': credit_limit,
    'utilization_rate': np.round(utilization_rate, 2),
    'delinquencies_30_plus': delinquencies,
    'loan_amount': loan_amount,
    'interest_rate': interest_rate,
    'is_default': is_default
})

# Сохраняем в папку data
os.makedirs('data', exist_ok=True)
df.to_csv('data/bank_loan_data.csv', index=False)
print(f"Успешно сгенерировано {n_samples} строк. Доля дефолтов: {df['is_default'].mean():.2%}")
