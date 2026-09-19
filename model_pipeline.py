import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import RandomForestRegressor

    class XGBRegressor(RandomForestRegressor):
        """Compatibility shim for environments without xgboost installed."""

        def __init__(
            self,
            n_estimators=120,
            learning_rate=0.06,
            max_depth=6,
            random_state=None,
            **kwargs
        ):
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            super().__init__(
                n_estimators=n_estimators,
                random_state=random_state,
                **kwargs,
            )

# Approximate airport coordinates for geospatial features
AIRPORT_COORDS = {
    'JFK': (40.6413, -73.7781),
    'LAX': (33.9425, -118.4081),
    'SFO': (37.6213, -122.3790),
    'ORD': (41.9742, -87.9073),
    'MIA': (25.7959, -80.2870),
    'LHR': (51.4700, -0.4543),
    'BOS': (42.3656, -71.0096),
    'NBO': (-1.3192, 36.9278),
    'JNB': (-26.2041, 28.0473),
    'ADD': (8.9779, 38.7993),
    'KGL': (-1.9674, 30.1395),
    'CPT': (-33.9698, 18.6021),
    'DAR': (-6.8781, 39.2026),
    'MBA': (-4.0348, 39.5942),
    'ABJ': (5.2614, -3.9263),
    'ACC': (5.6058, -0.1668),
    'LOS': (6.5774, 3.3212),
    'DXB': (25.2532, 55.3657),
    'AMS': (52.3105, 4.7683),
    'CDG': (49.0097, 2.5479),
    'FRA': (50.0379, 8.5622),
    'MAD': (40.4719, -3.5626),
    'IST': (41.2753, 28.7519)
}

def calculate_haversine(origin: str, dest: str) -> float:
    """Calculates distance between two airport codes in kilometers."""
    if origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
        return 1000.0  # Default fallback distance
    
    lat1, lon1 = AIRPORT_COORDS[origin]
    lat2, lon2 = AIRPORT_COORDS[dest]
    
    R = 6371.0 # Radius of Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_synthetic_dataset(n_samples: int = 4000) -> pd.DataFrame:
    """Generates synthetic historical fare observations in Kenyan shillings for African carriers and global routes."""
    np.random.seed(42)
    routes = [
        ('NBO', 'LHR'), ('NBO', 'JFK'), ('NBO', 'AMS'), ('NBO', 'DXB'),
        ('JNB', 'DXB'), ('JNB', 'LHR'), ('JNB', 'JFK'), ('JNB', 'AMS'),
        ('ADD', 'AMS'), ('ADD', 'DXB'), ('ADD', 'CDG'), ('ADD', 'MAD'),
        ('KGL', 'CDG'), ('KGL', 'FRA'), ('KGL', 'IST'), ('DAR', 'JNB'),
        ('MBA', 'LHR'), ('ABJ', 'CDG'), ('ACC', 'LHR'), ('LOS', 'DXB'),
        ('CPT', 'LHR'), ('CPT', 'JFK'), ('CPT', 'AMS'), ('JNB', 'MAD')
    ]
    airlines = [
        'Kenya Airways', 'Ethiopian Airlines', 'RwandAir', 'Airlink',
        'South African Airways', 'Air Tanzania', 'ASKY Air', 'Qatar Airways',
        'Emirates', 'FlySafair'
    ]

    base_fares = {
        ('NBO', 'LHR'): 56000, ('NBO', 'JFK'): 86000, ('NBO', 'AMS'): 49000, ('NBO', 'DXB'): 43000,
        ('JNB', 'DXB'): 42000, ('JNB', 'LHR'): 61000, ('JNB', 'JFK'): 82000, ('JNB', 'AMS'): 52000,
        ('ADD', 'AMS'): 47000, ('ADD', 'DXB'): 36000, ('ADD', 'CDG'): 50000, ('ADD', 'MAD'): 48000,
        ('KGL', 'CDG'): 51000, ('KGL', 'FRA'): 47000, ('KGL', 'IST'): 44000, ('DAR', 'JNB'): 33000,
        ('MBA', 'LHR'): 60000, ('ABJ', 'CDG'): 38000, ('ACC', 'LHR'): 55000, ('LOS', 'DXB'): 46000,
        ('CPT', 'LHR'): 59000, ('CPT', 'JFK'): 78000, ('CPT', 'AMS'): 51000, ('JNB', 'MAD'): 64000
    }

    records = []
    base_date = datetime(2026, 1, 1)

    for _ in range(n_samples):
        route = routes[np.random.choice(len(routes))]
        airline = np.random.choice(airlines)

        search_date = base_date + timedelta(days=int(np.random.randint(0, 120)))
        days_to_dep = int(np.random.randint(1, 60))
        dep_date = search_date + timedelta(days=days_to_dep)

        base_price = base_fares[route]

        airline_mult = {
            'Kenya Airways': 1.22,
            'Ethiopian Airlines': 1.18,
            'RwandAir': 1.12,
            'Airlink': 1.05,
            'South African Airways': 1.16,
            'Air Tanzania': 1.09,
            'ASKY Air': 1.08,
            'Qatar Airways': 1.28,
            'Emirates': 1.31,
            'FlySafair': 0.97
        }[airline]
        demand_curve = np.exp(-days_to_dep / 18.0) * 1.9 + 0.82
        weekend_mult = 1.22 if dep_date.weekday() in [4, 5, 6] else 1.0

        layovers = int(np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
        layover_discount = 1.0 - (layovers * 0.12)

        price = round(base_price * airline_mult * demand_curve * weekend_mult * layover_discount + np.random.normal(0, 1500), 2)

        pct_noise = np.random.normal(0.01, 0.07)
        if days_to_dep <= 14:
            pct_noise += 0.06

        future_price = round(price * (1 + pct_noise), 2)
        diff = (future_price - price) / price

        if diff > 0.03:
            trend = 'Rise'
        elif diff < -0.03:
            trend = 'Drop'
        else:
            trend = 'Stable'

        records.append({
            'search_date': search_date,
            'departure_date': dep_date,
            'origin': route[0],
            'destination': route[1],
            'airline': airline,
            'layovers': layovers,
            'days_until_flight': days_to_dep,
            'current_price': price,
            'future_price_7d': future_price,
            'price_trend_7d': trend
        })

    return pd.DataFrame(records)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates temporal, geospatial, and demand indicator features."""
    df = df.copy()
    
    # 1. Temporal Features
    df['search_date'] = pd.to_datetime(df['search_date'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    
    df['dep_day_of_week'] = df['departure_date'].dt.dayofweek
    df['dep_month'] = df['departure_date'].dt.month
    df['is_weekend_dep'] = df['dep_day_of_week'].apply(lambda x: 1 if x in [4, 5, 6] else 0)
    
    # Seasonality Flag (Summer / Holidays)
    df['is_high_season'] = df['dep_month'].apply(lambda m: 1 if m in [6, 7, 8, 12] else 0)
    
    # 2. Geospatial Distance
    df['route_distance_km'] = df.apply(
        lambda row: calculate_haversine(row['origin'], row['destination']), axis=1
    )
    
    # 3. Ratio / Interaction Features
    df['price_per_km'] = df['current_price'] / (df['route_distance_km'] + 1e-5)
    df['urgency_index'] = 1.0 / (df['days_until_flight'] + 1)
    
    return df

class TravelDealForecaster:
    """Dual ML Pipeline containing classification and regression models."""
    
    def __init__(self):
        self.num_cols = [
            'layovers', 'days_until_flight', 'current_price', 
            'dep_day_of_week', 'dep_month', 'is_weekend_dep', 
            'is_high_season', 'route_distance_km', 'price_per_km', 'urgency_index'
        ]
        self.cat_cols = ['origin', 'destination', 'airline']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)
            ]
        )
        
        self.classifier = Pipeline([
            ('prep', self.preprocessor),
            ('clf', RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42))
        ])
        
        self.regressor = Pipeline([
            ('prep', self.preprocessor),
            ('reg', XGBRegressor(n_estimators=120, learning_rate=0.06, max_depth=6, random_state=42))
        ])

    def train_with_time_split(self, df: pd.DataFrame):
        """Trains models using time-ordered cross-validation split."""
        df = engineer_features(df)
        df = df.sort_values('search_date').reset_index(drop=True)
        
        feature_cols = self.num_cols + self.cat_cols
        X = df[feature_cols]
        y_clf = df['price_trend_7d']
        y_reg = df['future_price_7d']
        
        # Sequential temporal split (80% train, 20% test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
        y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
        
        # Fit pipelines
        self.classifier.fit(X_train, y_clf_train)
        self.regressor.fit(X_train, y_reg_train)
        
        # Evaluating performance
        clf_preds = self.classifier.predict(X_test)
        reg_preds = self.regressor.predict(X_test)
        
        acc = accuracy_score(y_clf_test, clf_preds)
        mae = mean_absolute_error(y_reg_test, reg_preds)
        
        return {"accuracy": acc, "mae": mae, "test_samples": len(X_test)}

    def predict(self, single_row_df: pd.DataFrame) -> dict:
        """Runs predictions on single flight query dataframe."""
        df_feats = engineer_features(single_row_df)
        feature_cols = self.num_cols + self.cat_cols
        X = df_feats[feature_cols]
        
        trend = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probs) * 100)
        
        pred_future_price = float(self.regressor.predict(X)[0])
        
        return {
            'trend': trend,
            'confidence': confidence,
            'predicted_future_price': pred_future_price
        }

if __name__ == "__main__":
    df_raw = generate_synthetic_dataset(3000)
    forecaster = TravelDealForecaster()
    metrics = forecaster.train_with_time_split(df_raw)
    print(f"Training Complete. Classification Acc: {metrics['accuracy']:.2%}, Regression MAE: ${metrics['mae']:.2f}")