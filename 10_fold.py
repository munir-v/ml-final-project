import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV

df = pd.read_csv("data/cleaned/final_data.csv")
df.drop(columns=["calendarDate"], inplace=True)

# Handling nan and inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# Select features and variable to predict
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

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using recursive feature elimination with cross-validation
estimator = RandomForestRegressor(random_state=47)
selector = RFECV(estimator, step=1, cv=KFold(5), scoring="neg_mean_squared_error")
selector = selector.fit(X_scaled, y)

# Print the selected features
selected_features = X.columns[selector.support_]
print(f"Selected features: {selected_features}")

# Models to try
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(random_state=47),
    "Ridge Regression": Ridge(random_state=47),
    "Elastic Net": ElasticNet(random_state=47),
    "Support Vector Regression": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=47),
    "Random Forest": RandomForestRegressor(random_state=47),
    "Gradient Boosting": GradientBoostingRegressor(random_state=47),
}

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=47)
for name, model in models.items():
    scores = cross_val_score(
        model,
        X_scaled[:, selector.support_],
        y,
        cv=kf,
        scoring="neg_mean_squared_error",
    )
    r2_scores = cross_val_score(
        model, X_scaled[:, selector.support_], y, cv=kf, scoring="r2"
    )
    print(f"{name} - Cross-Validation Results:")
    for i, (mse, r2) in enumerate(zip(scores, r2_scores)):
        print(f"  Fold {i + 1}: MSE = {-mse}, R^2 = {r2}")
    print(f"Average MSE = {-np.mean(scores)}, Average R^2 = {np.mean(r2_scores)}\n")
