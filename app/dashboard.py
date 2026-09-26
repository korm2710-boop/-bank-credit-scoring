# Интерактивное приложение
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Настройка страницы
st.set_page_config(page_title="T-Bank Credit Scoring Dashboard", layout="wide")
st.title("📊 Оптимизация бизнес-метрик кредитного скоринга")
st.subheader("Кейс для Департамента Рисков Т-Банка")

# 2. Кэширование загрузки данных и обучения модели (чтобы дашборд не тормозил)
@st.cache_data
def load_and_train():
    df = pd.read_csv('data/bank_loan_data_features.csv')
    X = df[['age', 'monthly_income', 'credit_limit', 'utilization_rate', 'delinquencies_30_plus']]
    y = df['is_default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_financials = df.loc[y_test.index].copy()
    test_financials['prob_default'] = y_pred_proba
    
    return test_financials

test_financials = load_and_train()

# 3. Боковая панель управления (Интерфейс бизнес-аналитика)
st.sidebar.header("⚙️ Экономические параметры")
st.sidebar.markdown("Настройте параметры кредитного продукта для пересчета P&L:")

# Позволяем пользователю менять логику доходности и потерь
cost_of_risk_multiplier = st.sidebar.slider("Коэффициент тяжести потерь (LGD)", 0.5, 1.5, 1.0, 0.1)
bonus_interest = st.sidebar.slider("Надбавка к процентной ставке (годовых)", -0.05, 0.10, 0.00, 0.01)

# 4. Расчет финансового эффекта в реальном времени
thresholds = np.linspace(0.01, 0.99, 100)
profits = []

for t in thresholds:
    decision = np.where(test_financials['prob_default'] < t, 1, 0)
    
    # Доход с учетом корректировки пользователем
    adjusted_rate = test_financials['interest_rate'] + bonus_interest
    interest_gain = np.sum(((decision == 1) & (test_financials['is_default'] == 0)) * (test_financials['loan_amount'] * adjusted_rate))
    
    # Потери с учетом корректировки тяжести риска
    loan_loss = np.sum(((decision == 1) & (test_financials['is_default'] == 1)) * (test_financials['loan_amount'] * cost_of_risk_multiplier))
    
    profits.append(interest_gain - loan_loss)

best_idx = np.argmax(profits)
best_threshold = thresholds[best_idx]
max_profit = profits[best_idx]
p_at_50 = profits[np.abs(thresholds - 0.5).argmin()]
additional_money = max_profit - p_at_50

# 5. Вывод ключевых метрик (Метрики верхнего уровня)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Оптимальный порог одобрения", value=f"{best_threshold:.2f}")
with col2:
    st.metric(label="Макс. прибыль портфеля", value=f"{max_profit:,.0f} ₽")
with col3:
    st.metric(label="Эффект от оптимизации (vs порог 0.5)", value=f"+ {additional_money:,.0f} ₽", delta=f"{additional_money:,.0f} ₽")

# 6. Отрисовка интерактивного графика
st.write("### График зависимости прибыли банка от порога отсечения моделей")
chart_data = pd.DataFrame({
    'Порог одобрения': thresholds,
    'Прибыль (₽)': profits
}).set_index('Порог одобрения')

st.line_chart(chart_data)

st.markdown("""
**Как читать этот график:** 
* Каждая точка на линии — это финансовый результат банка при выборе определенной строгости модели.
* Проект доказывает: максимизация стандартных ML-метрик (с порогом 0.5) приводит к упущенной выгоде. Оптимальный порог, рассчитанный выше, максимизирует **чистый процентный доход** банка.
""")
