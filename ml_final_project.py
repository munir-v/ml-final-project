import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/cleaned/final_data.csv")
df.drop(columns=["calendarDate"], inplace=True)

# Check for NaN values
if df.isna().sum().sum() > 0:
    # Handling NaN values - fill with the median
    df.fillna(df.median(), inplace=True)

# Check for Inf/-Inf values
if not np.all(np.isfinite(df)):
    # Replace Inf/-Inf with NaN, then handle NaNs as above
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)


# Select features and target variable
X = df[
    [
        "totalSteps",
        "highlyActiveSeconds",
        "moderateIntensityMinutes",
        "vigorousIntensityMinutes",
        "minHeartRate",
        "maxHeartRate",
        "restingHeartRate",
        "currentDayRestingHeartRate",
        "hrvWeeklyAverage",
        "deepSleepHours",
        "lightSleepHours",
        "remSleepHours",
        "awakeSleepHours",
        "totalSleepHours",
        # "overallScore",
        # "qualityScore",
    ]
]
y = df["overallScore"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
