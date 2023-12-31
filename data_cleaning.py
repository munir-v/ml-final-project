import pandas as pd
from pandas import json_normalize

hrv1 = pd.read_json("/content/TrainingReadinessDTO_20221128_20230308_107145992.json")
hrv2 = pd.read_json("/content/TrainingReadinessDTO_20230308_20230616_107145992.json")
hrv3 = pd.read_json("/content/TrainingReadinessDTO_20230616_20230924_107145992.json")
hrv4 = pd.read_json("/content/TrainingReadinessDTO_20230924_20240102_107145992.json")
merged_hrv_data = pd.concat([hrv1, hrv2, hrv3, hrv4])

test = merged_hrv_data[["calendarDate", "hrvWeeklyAverage"]]

activity1 = pd.read_json("/content/UDSFile_2022-10-07_2023-01-15.json")
activity2 = pd.read_json("/content/UDSFile_2023-01-15_2023-04-25.json")
activity3 = pd.read_json("/content/UDSFile_2023-04-25_2023-08-03.json")
activity4 = pd.read_json("/content/UDSFile_2023-08-03_2023-11-11.json")
merged_activity_data = pd.concat([activity1, activity2, activity3, activity4])

merged_activity_data = pd.merge(
    merged_activity_data, merged_hrv_data, on="calendarDate", how="inner"
)
merged_activity_data.to_csv("activity_data.csv", index=False)

pd.set_option("display.max_columns", None)
data = pd.read_csv("activity_data.csv")
data.columns
data.head()

data = data[
    [
        "calendarDate",
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
        "restingHeartRate",
        "currentDayRestingHeartRate",
        "hrvWeeklyAverage",
    ]
]
data.dropna()
data.to_csv("cleaned_activity_data.csv", index=False)

sleep1 = pd.read_json("/content/2023-08-03_2023-11-11_107145992_sleepData.json")
sleep2 = pd.read_json("/content/2023-01-15_2023-04-25_107145992_sleepData.json")
sleep3 = pd.read_json("/content/2023-04-25_2023-08-03_107145992_sleepData.json")
sleep4 = pd.read_json("/content/2022-10-07_2023-01-15_107145992_sleepData.json")
merged_sleep_data = pd.concat([sleep1, sleep2, sleep3, sleep4])

sleep_scores_expanded = json_normalize(merged_sleep_data["sleepScores"])
merged_sleep_data = merged_sleep_data.drop("sleepScores", axis=1)
sleep = merged_sleep_data.join(sleep_scores_expanded)
sleep.head()

sleep["deepSleepSeconds"] = pd.to_timedelta(sleep["deepSleepSeconds"], unit="S")
sleep[["lightSleepSeconds", "remSleepSeconds", "awakeSleepSeconds"]] = sleep[
    ["lightSleepSeconds", "remSleepSeconds", "awakeSleepSeconds"]
].apply(lambda x: pd.to_timedelta(x, unit="S"))

sleep["deepSleepHours"] = sleep["deepSleepSeconds"].dt.total_seconds() / 3600
sleep["lightSleepHours"] = sleep["lightSleepSeconds"].dt.total_seconds() / 3600
sleep["remSleepHours"] = sleep["remSleepSeconds"].dt.total_seconds() / 3600
sleep["awakeSleepHours"] = sleep["awakeSleepSeconds"].dt.total_seconds() / 3600
sleep["totalSleepHours"] = (
    sleep["deepSleepHours"]
    + sleep["lightSleepHours"]
    + sleep["remSleepHours"]
    + sleep["awakeSleepHours"]
)

sleep = sleep[
    [
        "calendarDate",
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

sleep.to_csv("cleaned_activity_data.csv", index=False)

all_data = pd.merge(data, sleep, on="calendarDate", how="inner")
all_data.dropna()
all_data.to_csv("final_data.csv", index=False)
# all_data
