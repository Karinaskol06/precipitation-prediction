"""
Модуль для отримання даних з Open Meteo API
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import os
from typing import Optional

class WeatherDataFetcher:
    """
    Клас для завантаження історичних даних погоди з Open-Meteo
    """
    
    def __init__(self, latitude: float = 49.84, longitude: float = 24.03):
        """
        Ініціалізація з координатами (за замовчуванням - Львів)
        """
        self.latitude = latitude
        self.longitude = longitude
        
        # Дані, які вже були отримані, будуть братись з кешу
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        # Повторні спроби (якщо сервер перевантажений)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
        

    def fetch_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Отримує щоденні дані погоди за вказаний період
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Перевіряємо, чи період не надто довгий (Open-Meteo може мати обмеження)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days_diff = (end - start).days
        
        if days_diff > 730:  # більше 2 років
            print(f"Період {days_diff} днів може бути занадто довгим. Розбиваємо на частини...")
            return self._fetch_data_in_chunks(start_date, end_date)
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "precipitation_sum",
                "rain_sum",
                "temperature_2m_max",
                "temperature_2m_min",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
                "sunshine_duration"
            ],
            "timezone": "Europe/Kiev"
        }
        
        try:
            # Робимо запит до API
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Обробляємо щоденні дані
            daily = response.Daily()
            
            # Отримуємо всі змінні у вигляді numpy масивів 
            # Вони швидше і зручніше для обробки 
            daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()
            daily_rain_sum = daily.Variables(1).ValuesAsNumpy()
            daily_temperature_2m_max = daily.Variables(2).ValuesAsNumpy()
            daily_temperature_2m_min = daily.Variables(3).ValuesAsNumpy()
            daily_wind_speed_10m_max = daily.Variables(4).ValuesAsNumpy()
            daily_shortwave_radiation_sum = daily.Variables(5).ValuesAsNumpy()
            daily_sunshine_duration = daily.Variables(6).ValuesAsNumpy()
            
            # Створюємо дати для кожного дня на основі часу початку і кінця та інтервалу
            # Конвертуємо час з секунд в datetime за допомогою pandas
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                )
            }
            
            # Додаємо всі змінні
            daily_data["precipitation_sum"] = daily_precipitation_sum
            daily_data["rain_sum"] = daily_rain_sum
            daily_data["temp_max"] = daily_temperature_2m_max
            daily_data["temp_min"] = daily_temperature_2m_min
            daily_data["wind_speed_max"] = daily_wind_speed_10m_max
            daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
            daily_data["sunshine_duration"] = daily_sunshine_duration
            
            # Конвертуємо тривалість сонячного сяйва з секунд в години
            daily_data["sunshine_hours"] = daily_sunshine_duration / 3600
            
            # Створюємо DataFrame
            df = pd.DataFrame(data=daily_data)
            
            # Конвертуємо дату в datetime
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            return df
            
        except Exception as e:
            print(f"Помилка при отриманні даних: {e}")
            return pd.DataFrame()
    

    def save_to_csv(self, df: pd.DataFrame, filename: str = "weather_daily.csv") -> str:
        """
        Зберігає дані у CSV файл
        """
        os.makedirs("data/raw", exist_ok=True)
        filepath = os.path.join("data/raw", filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Завантажує дані з CSV файлу
        """
        return pd.read_csv(filepath, parse_dates=['date'])