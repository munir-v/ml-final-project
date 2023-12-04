import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV

df = pd.read_csv("data/cleaned/final_data.csv")
df.drop(columns=["calendarDate"], inplace=True)

# Handling NaN and Inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# Select features and target variable
X = df[
    [
        "totalSteps",
        "highlyActiveSeconds",
        "activeSeconds",
        "moderateIntensityMinutes",
        "minAvgHeartRate",
        "maxAvgHeartRate",
        "moderateIntensityMinutes",
        "vigorousIntensityMinutes",
        "minHeartRate",
        "maxHeartRate",
        # "restingHeartRate",
        # "currentDayRestingHeartRate",
        "hrvWeeklyAverage",
        "deepSleepHours",
        "lightSleepHours",
        "remSleepHours",
        "awakeSleepHours",
        "totalSleepHours",
        "overallScore",
        "qualityScore",
        "averageRespiration",
        "lowestRespiration",
        "highestRespiration",
        "awakeCount",
        "avgSleepStress",
        "restlessMomentCount",
    ]
]
y = df["currentDayRestingHeartRate"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=47
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using RFECV
estimator = RandomForestRegressor(random_state=47)
selector = RFECV(estimator, step=1, cv=KFold(5), scoring="neg_mean_squared_error")
selector = selector.fit(X_train_scaled, y_train)

# Print the selected features
selected_features = X.columns[selector.support_]
print(f"Selected features: {selected_features}")

# Train and evaluate models with selected features
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=47),
    "Random Forest": RandomForestRegressor(random_state=47),
    "Gradient Boosting": GradientBoostingRegressor(random_state=47),
}

for name, model in models.items():
    model.fit(X_train_scaled[:, selector.support_], y_train)
    y_pred = model.predict(X_test_scaled[:, selector.support_])
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse}, R^2 Score: {r2}")
