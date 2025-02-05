import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math

file_path = 'Fantasy_Football_Stats_10.csv'  
df = pd.read_csv(file_path)

pd.set_option('display.max_columns', None)


df = df.sort_values(by=['PLAYER', 'SEASON'])

for stat in ['FPTS', 'CMP', 'ATT', 'CMP%', 'YDS', 'AVG', 'TD', 'INT', 'RATING', 'ATT_RUSH', 'YDS_RUSH', 'AVG_RUSH', 'TD_RUSH']:
    df[f'{stat}_prev_season'] = df.groupby('PLAYER')[stat].shift(1)  # Last season's value
    df[f'{stat}_2yr_avg'] = df.groupby('PLAYER')[stat].rolling(2).mean().reset_index(level=0, drop=True)  # Use last 2 years 
    df[f'{stat}_trend'] = df[f'{stat}_prev_season'] - df[f'{stat}_2yr_avg']  # Performance trend

df = df.dropna()

features = df.drop(['SEASON', 'TEAM', 'PLAYER', 'FPTS'], axis=1)
target = df['FPTS']

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_estimators=100)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, features, target, cv=cv, scoring='neg_mean_squared_error')
cv_rmse_scores = [math.sqrt(abs(score)) for score in cv_scores]

print("\nCross-Validation RMSE scores:", [f"{rmse:.2f}" for rmse in cv_rmse_scores])
print(f"Average Cross-Validation RMSE: {sum(cv_rmse_scores) / len(cv_rmse_scores):.2f}")

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))

y_test_pred = model.predict(X_test)
test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nTraining RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")

def predict_future_fpts(player_name):
    player_data = df[df['PLAYER'] == player_name].sort_values(by='SEASON', ascending=False).head(1)

    if player_data.empty:
        print(f"No data found for {player_name}.")
        return

    player_features = player_data.drop(['SEASON', 'TEAM', 'PLAYER', 'FPTS'], axis=1)

    predicted_fpts = model.predict(player_features)[0]
    print(f"Predicted 2025 Fantasy Points for {player_name}: {predicted_fpts:.2f}")

# Predict for specific players
predict_future_fpts("Lamar Jackson")
predict_future_fpts("Josh Allen")
predict_future_fpts("Jalen Hurts")
predict_future_fpts("Kyler Murray")
