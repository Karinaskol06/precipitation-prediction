"""
Модуль для прогнозування опадів на 5 днів.
Використовує алгоритми Random Forest або Logistic Regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os


class RainPredictor:
    """Прогнозування опадів на основі погодних даних"""
    
    def __init__(self, model_type: str = "random_forest", forecast_days: int = 5):
        self.model_type = model_type
        self.forecast_days = forecast_days
        self.models = []
        self.scalers = []
        self.feature_names = None
        

    def prepare_features(self, df: pd.DataFrame):
        """
        Готує ознаки та цільові змінні для кожного дня прогнозу
        """
        data = df.copy().sort_values('date').reset_index(drop=True).dropna()
        
        if len(data) < 20:
            raise ValueError(f"Недостатньо даних: {len(data)} днів, потрібно мінімум 20")
        
        # Х - ознаки (дані погоди), у - прогноз (1 - буде дощ, 0 - не буде)
        X_list, y_list = [], []
        
        for day in range(1, self.forecast_days + 1):
            # Цільова змінна (те, що хочемо передбачити): опади через day днів
            target = (data['precipitation_sum'].shift(-day) > 0).astype(int)
            valid_data = data.dropna(subset=[target.name]).copy()
            target = target.dropna()
            
            features = []
            for idx in range(len(valid_data)):
                curr = valid_data.iloc[idx]
                
                # Базові ознаки
                feat = [
                    curr['temp_max'], curr['temp_min'], curr['wind_speed_max'],
                    curr['precipitation_sum'], curr['rain_sum'],
                    curr['shortwave_radiation_sum'], curr['sunshine_hours']
                ]
                
                # Ковзні середні за 3 дні
                if idx >= 3:
                    last3 = valid_data.iloc[idx-3:idx]
                    feat.extend([
                        last3['precipitation_sum'].mean(),
                        last3['temp_max'].mean(),
                        (last3['precipitation_sum'] > 0).sum(),
                        last3['wind_speed_max'].mean()
                    ])
                else:
                    feat.extend([
                        curr['precipitation_sum'],
                        curr['temp_max'],
                        1 if curr['precipitation_sum'] > 0 else 0,
                        curr['wind_speed_max']
                    ])
                
                # Сезонність
                date = pd.to_datetime(curr['date'])
                feat.extend([date.month, date.dayofyear])
                
                features.append(feat)
            
            X_list.append(np.array(features, dtype=np.float32))
            y_list.append(target.values.astype(np.int32))
        
        self.feature_names = [
            'temp_max', 'temp_min', 'wind_speed', 'precip', 'rain',
            'radiation', 'sunshine_hours',
            # Ознаки за 3 останні дні (ковзні середні)
            'precip_3d_mean', 'temp_3d_mean', 'rainy_days_3d', 'wind_3d_mean',
            'month', 'day_of_year'
        ]
        
        return X_list, y_list
    

    def train(self, X_list, y_list, test_size=0.2):
        """
        Навчає окремі моделі для кожного дня
        """
        self.models = []
        self.scalers = []
        all_metrics = []
        
        for day in range(self.forecast_days):
            X, y = X_list[day], y_list[day]
            
            # Мінімальна кількість даних для навчання - 10 (для пошуку патернів)
            if len(X) < 10:
                continue
            
            # Розбиття даних на тренувальні та тестові набори
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Масштабування
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) # Навчаємо на тренувальних даних
            X_test = scaler.transform(X_test) # Застосовуємо ті ж параметри до тестових
            
            # Модель - Random Forest або Logistic Regression
            if self.model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=200, # 200 дерев 
                    max_depth=15, # глибина дерева
                    min_samples_split=2, # мінімум зразків для розгалуження
                    random_state=42
                )
            else:
                model = LogisticRegression(
                    random_state=42, 
                    max_iter=1000, # максимум ітерацій 
                    class_weight='balanced' # баланс класів для кращої роботи з дисбалансом
                )
            
            # Навчаємо модель і робимо прогноз
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Метрики
            metrics = {
                'day': day + 1,
                # точність прогнозу (доля правильних відповідей) з тестових даних
                'accuracy': accuracy_score(y_test, y_pred),
                # доля правильних позитивних прогнозів серед усіх позитивних прогнозів
                'precision': precision_score(y_test, y_pred, zero_division=0),
                # повнота (доля правильних позитивних прогнозів серед усіх фактичних позитивів)
                'recall': recall_score(y_test, y_pred, zero_division=0),
                # баланс між точністю і повнотою виявлення
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            self.models.append(model)
            self.scalers.append(scaler)
            all_metrics.append(metrics)
        
        if not self.models:
            raise ValueError("Не вдалося навчити жодної моделі")
        
        return all_metrics
    

    def predict(self, features):
        """
        Прогноз на основі ознак
        """
        if not self.models:
            raise ValueError("Моделі не навчені")
        
        # Якщо вхідні дані - один набір ознак, перетворюємо його в 2D масив для сумісності з моделями
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Прогноз для кожного дня
        predictions = []
        for i in range(len(self.models)):
            X_scaled = self.scalers[i].transform(features)
            pred = self.models[i].predict(X_scaled)[0]
            prob = self.models[i].predict_proba(X_scaled)[0]
            prob = prob[pred] if len(prob) > 1 else (1.0 if pred == 1 else 0.0)
            predictions.append((pred, prob))
        
        # Доповнення до forecast_days 
        while len(predictions) < self.forecast_days:
            predictions.append(predictions[-1] if predictions else (0, 0.5))
        
        return predictions[:self.forecast_days]
    

    def predict_future(self, last_days):
        """
        Прогноз на основі останніх 4 днів даних
        """
        if len(last_days) < 4:
            raise ValueError("Потрібно мінімум 4 дні даних")
        
        # Використовуємо останній день для поточних ознак і 3 попередні для ковзних середніх
        last = last_days.iloc[-1]
        last3 = last_days.iloc[-4:-1]
        date = pd.to_datetime(last['date'])
        
        features = np.array([[
            last['temp_max'], last['temp_min'], last['wind_speed_max'],
            last['precipitation_sum'], last['rain_sum'],
            last['shortwave_radiation_sum'], last['sunshine_hours'],
            last3['precipitation_sum'].mean(),
            last3['temp_max'].mean(),
            (last3['precipitation_sum'] > 0).sum(),
            last3['wind_speed_max'].mean(),
            date.month, date.dayofyear
        ]], dtype=np.float32)
        
        return self.predict(features)
    
    # Збереження та завантаження моделей
    def save(self, path="models/rain_predictor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'forecast_days': self.forecast_days
        }, path)
    
    def load(self, path="models/rain_predictor.pkl"):
        data = joblib.load(path)
        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.forecast_days = data['forecast_days']