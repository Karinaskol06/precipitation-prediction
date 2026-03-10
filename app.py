"""
Головний Streamlit застосунок для прогнозування опадів
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_fetch import WeatherDataFetcher
from src.model_trainer import RainPredictor

# Налаштування сторінки
st.set_page_config(
    page_title="Прогноз опадів у Львові",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ Прогноз опадів для м. Львів на основі даних Open-Meteo")
st.markdown("---")

with st.sidebar:
    st.header("Налаштування")
    st.info(f"📍 Львів (49.84°N, 24.03°E)")
    lat, lon = 49.84, 24.03
    
    st.subheader("📅 Період даних")
    days = st.slider("Кількість днів для завантаження", 60, 730, 365)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    st.write(f"Період: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")
 
    st.subheader("Виберіть модель ML")
    model_type = st.radio(
        "Оберіть алгоритм",
        ["random_forest", "logistic_regression"],
        format_func=lambda x: "Random Forest" if x == "random_forest" else "Логістична регресія"
    )
    
    st.subheader("Прогноз")
    forecast_days = st.slider("На скільки днів зробити прогноз", 1, 7, 5)
    
    st.markdown("---")
    fetch_button = st.button("1. Отримати дані з Open-Meteo", use_container_width=True, type="primary")
    train_button = st.button("2. Навчити модель", use_container_width=True, type="primary")
    predict_button = st.button("3. Зробити прогноз", use_container_width=True, type="primary")

# Основний контейнер для виводу результатів
main_container = st.container()

with main_container:
    # Ініціалізація сесійних змінних для збереження даних, моделі та метрик між взаємодіями
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'fetcher' not in st.session_state:
        st.session_state.fetcher = WeatherDataFetcher(latitude=lat, longitude=lon)
    
    # 1. Отримання даних
    if fetch_button:
        with st.spinner("Завантаження даних з Open-Meteo..."):
            try:
                df = st.session_state.fetcher.fetch_daily_data(
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                
                if not df.empty:
                    st.session_state.weather_data = df
                    
                    # Збереження результатів у CSV файл
                    csv_path = st.session_state.fetcher.save_to_csv(df, f"lviv_weather_{days}days.csv")
                    
                    st.success(f"Дані успішно завантажено! Отримано {len(df)} днів")
                    st.subheader("Зразок даних")
                    st.dataframe(df.head(10))
                    
                    # Базова статистика
                    st.subheader("Статистика опадів")
                    col1, col2, col3 = st.columns(3)
                    
                    rainy_days = (df['precipitation_sum'] > 0).sum()
                    total_days = len(df)
                    
                    col1.metric("Дні з опадами", f"{rainy_days} днів")
                    col2.metric("Дні без опадів", f"{total_days - rainy_days} днів")
                    col3.metric("Відсоток опадів", f"{(rainy_days/total_days*100):.1f}%")
                    
                    # Налаштування стилю для темного фону
                    plt.style.use('dark_background')

                    # Графік опадів (щоденні суми) зроблений з matplotlib
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#00162a')  
                    ax.set_facecolor('#364573')  

                    ax.bar(df['date'], df['precipitation_sum'], color='#F2CA50', alpha=0.7, label='Опади')
                    ax.set_xlabel('Дата', color='white')
                    ax.set_ylabel('Опади (мм)', color='white')
                    ax.set_title('Щоденні опади за період', color='white', fontsize=14, fontweight='bold')
                    ax.tick_params(colors='white') 
                    ax.legend(facecolor='#404040', labelcolor='white', framealpha=0.8)
                    ax.grid(True, alpha=0.2, color='white')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Графік температур
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    fig2.patch.set_facecolor('#00162a') 
                    ax2.set_facecolor('#364573') 

                    ax2.plot(df['date'], df['temp_max'], color='#73062E', label='Макс температура', linewidth=2.5)
                    ax2.plot(df['date'], df['temp_min'], color='#F2CA50', label='Мін температура', linewidth=2.5)
                    ax2.fill_between(df['date'], df['temp_min'], df['temp_max'], alpha=0.2, color='#73062E')  

                    ax2.set_xlabel('Дата', color='white')
                    ax2.set_ylabel('Температура (°C)', color='white')
                    ax2.set_title('Температура за період', color='white', fontsize=14, fontweight='bold')
                    ax2.tick_params(colors='white')
                    ax2.legend(facecolor='#404040', labelcolor='white', framealpha=0.8)
                    ax2.grid(True, alpha=0.2, color='white')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                else:
                    st.error("Не вдалося завантажити дані. Перевірте з'єднання.")
            except Exception as e:
                st.error(f"Помилка: {str(e)}")
    
    # 2. Навчання моделі
    if train_button:
        if st.session_state.weather_data is None:
            st.warning("Спочатку завантажте дані!")
        else:
            with st.spinner(f"Навчання {forecast_days} моделей... (це може зайняти кілька секунд)"):
                try:
                    # Підготовка даних з вказаною кількістю днів для прогнозу
                    predictor = RainPredictor(model_type=model_type, forecast_days=forecast_days)
                    X_list, y_list = predictor.prepare_features(st.session_state.weather_data)
                    
                    # Навчання
                    metrics_list = predictor.train(X_list, y_list)
                    
                    st.session_state.model = predictor
                    st.session_state.metrics = metrics_list
                    st.success(f"Успішно навчено {len(metrics_list)} моделей!")
                    
                    # Показуємо метрики для кожного дня
                    st.subheader(f"Метрики якості для прогнозу на {forecast_days} днів")
                    
                    # Створюємо таблицю з метриками
                    metrics_df = pd.DataFrame(metrics_list)
                    st.dataframe(metrics_df[['day', 'accuracy', 'precision', 'recall', 'f1']].round(3))
                    
                    # Графік метрик
                    fig, ax = plt.subplots(figsize=(10, 5))
                    metrics_df.plot(x='day', y=['accuracy', 'precision', 'recall', 'f1'], 
                                   marker='o', ax=ax)
                    fig.patch.set_facecolor('#00162a')
                    ax.set_facecolor('#00162a')
                    ax.set_xlabel('День прогнозу')
                    ax.set_ylabel('Значення метрики')
                    ax.set_title('Метрики моделей для різних днів')
                    ax.grid(True, alpha=0.3)
                    ax.legend(['Точність', 'Precision', 'Recall', 'F1'])
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Помилка при навчанні: {str(e)}")
                    import traceback
                    st.exception(e)
    
    # 3. Прогноз
    if predict_button:
        if st.session_state.model is None:
            st.warning("Спочатку навчіть модель!")
        elif st.session_state.weather_data is None:
            st.warning("Спочатку завантажте дані!")
        else:
            st.subheader(f"Прогноз опадів на {st.session_state.model.forecast_days} днів")
            
            try:
                data = st.session_state.weather_data.sort_values('date')
                last_days = data.iloc[-4:]
                
                # Прогноз на вказану кількість днів
                predictions = st.session_state.model.predict_future(last_days)
                
                cols = st.columns(st.session_state.model.forecast_days)
                
                for day, (col, (pred, prob)) in enumerate(zip(cols, predictions), 1):
                    with col:
                        forecast_date = datetime.now() + timedelta(days=day)
                        st.markdown(f"### 📅 {forecast_date.strftime('%d.%m')}")
                        
                        if pred == 1:
                            st.error(f"🌧️ **Дощ**")
                        else:
                            st.success(f"☀️ **Сухо**")
                        
                        # Визначаємо колір для ймовірності
                        if prob > 0.8:
                            confidence = "🟢 Висока"
                        elif prob > 0.6:
                            confidence = "🟡 Середня"
                        else:
                            confidence = "🟠 Низька"
                        
                        st.metric("Ймовірність", f"{prob*100:.1f}%")
                        st.caption(f"Впевненість: {confidence}")
                
                # Детальна інформація
                with st.expander("Деталі прогнозу та поточні дані"):
                    st.write("**Погода сьогодні (останній день у даних):**")
                    last_day = last_days.iloc[-1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"- 🌡️ Температура макс: {last_day['temp_max']:.1f}°C")
                        st.write(f"- 🌡️ Температура мін: {last_day['temp_min']:.1f}°C")
                        st.write(f"- 💨 Вітер: {last_day['wind_speed_max']:.1f} км/год")
                    with col2:
                        st.write(f"- ☔ Опади сьогодні: {last_day['precipitation_sum']:.1f} мм")
                        st.write(f"- ☀️ Сонячних годин: {last_day['sunshine_hours']:.1f} год")
                        st.write(f"- ☀️ Сонячна радіація: {last_day['shortwave_radiation_sum']:.1f} MJ/m²")
                    
                    st.write("**Середнє за останні 3 дні:**")
                    last_3_days = last_days.iloc[-4:-1]
                    st.write(f"- ☔ Опади: {last_3_days['precipitation_sum'].mean():.1f} мм")
                    st.write(f"- 🌡️ Температура: {last_3_days['temp_max'].mean():.1f}°C")
                    st.write(f"- 🌧️ Днів з опадами: {(last_3_days['precipitation_sum'] > 0).sum()} з 3")
                    
                    # Таблиця з прогнозами
                    st.write("**Детальний прогноз по днях:**")
                    forecast_df = pd.DataFrame([
                        {
                            'День': i+1,
                            'Дата': (datetime.now() + timedelta(days=i+1)).strftime('%d.%m.%Y'),
                            'Прогноз': '🌧️ Дощ' if pred == 1 else '☀️ Без дощу',
                            'Ймовірність': f"{prob*100:.1f}%"
                        }
                        for i, (pred, prob) in enumerate(predictions)
                    ])
                    st.dataframe(forecast_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Помилка при прогнозуванні: {str(e)}")
                st.exception(e)

# Додаткова інформація внизу
st.markdown("---")
with st.expander("Про проєкт"):
    st.markdown("""
    ### Прогноз опадів на основі даних Open-Meteo
    
    **Джерело даних**: [Open-Meteo](https://open-meteo.com/) - безкоштовне API
    
    **Місто**: Львів (49.84°N, 24.03°E)
    
    **Модель ML**: 
    - Random Forest Classifier або Logistic Regression з scikit-learn
    - Цільова змінна: наявність опадів (precipitation_sum > 0)
    - 13 ознак: температура, вітер, опади, сонячна радіація, ковзні середні, сезонність
    - Окремі моделі для кожного дня прогнозу
    
    **Метрики**: accuracy, precision, recall, F1-score
    - accuracy: доля правильних відповідей серед усіх прогнозів
    - precision: доля правильних позитивних прогнозів серед усіх позитивних прогнозів
    - recall: доля правильних позитивних прогнозів серед усіх фактичних позитивів
    - F1-score: гармонічне середнє між правильністю та повнотою виявлення
    
    **Вимоги до даних**: мінімум 30 днів для якісного навчання
    
    **Як користуватися**:
    1. Натисніть "Отримати дані" - завантажаться дані з Open-Meteo
    2. Натисніть "Навчити модель" - навчаться 5 моделей (по одній на кожен день)
    3. Натисніть "Зробити прогноз" - побачите прогноз на 5 днів вперед
    
    **Автор**: Сколібог Каріна, 2026
    """)

# Кнопка для очищення
if st.sidebar.button("Очистити всі дані", use_container_width=True):
    st.session_state.weather_data = None
    st.session_state.model = None
    st.session_state.metrics = None
    st.rerun()