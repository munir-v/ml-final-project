import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV

df = pd.read_csv("data/cleaned/final_data.csv")
df.drop(columns=["calendarDate"], inplace=True)

# Handling NaN and Inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# Select features and target variable
X = df[["totalSteps", "highlyActiveSeconds", "activeSeconds", "moderateIntensityMinutes", "minAvgHeartRate", "maxAvgHeartRate", "moderateIntensityMinutes", "vigorousIntensityMinutes", "minHeartRate", "maxHeartRate", "hrvWeeklyAverage", "deepSleepHours", "lightSleepHours", "remSleepHours", "awakeSleepHours", "totalSleepHours", "overallScore", "qualityScore", "averageRespiration", "lowestRespiration", "highestRespiration", "awakeCount", "avgSleepStress", "restlessMomentCount"]]
y = df["currentDayRestingHeartRate"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using RFECV
estimator = RandomForestRegressor(random_state=47)
selector = RFECV(estimator, step=1, cv=KFold(5), scoring="neg_mean_squared_error")
selector = selector.fit(X_scaled, y)

# Print the selected features
selected_features = X.columns[selector.support_]
print(f"Selected features: {selected_features}")

# Define models
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

# Perform 10-Fold Cross-Validation and Store R^2 Values
kf = KFold(n_splits=10, shuffle=True, random_state=47)
model_r2_scores = {}

for name, model in models.items():
    r2_scores = cross_val_score(model, X_scaled[:, selector.support_], y, cv=kf, scoring='r2')
    model_r2_scores[name] = r2_scores
    print(f"{name} - Average R^2: {np.mean(r2_scores)}")

# Plotting the R^2 values
plt.figure(figsize=(15, 10))
for name, r2_scores in model_r2_scores.items():
    plt.plot(r2_scores, label=name)
plt.title('R^2 Scores Across 10 Folds for Different Models')
plt.xlabel('Fold Number')
plt.ylabel('R^2 Score')
plt.legend()
plt.show()
