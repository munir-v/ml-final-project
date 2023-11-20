import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import json_normalize

sleep1=pd.read_json('/content/2023-08-03_2023-11-11_107145992_sleepData.json')
sleep2=pd.read_json('/content/2023-01-15_2023-04-25_107145992_sleepData.json')
sleep3=pd.read_json('/content/2023-04-25_2023-08-03_107145992_sleepData.json')
sleep4=pd.read_json('/content/2022-10-07_2023-01-15_107145992_sleepData.json')
merged_sleep_data = pd.concat([sleep1, sleep2, sleep3, sleep4])

sleep_scores_expanded = json_normalize(merged_sleep_data['sleepScores'])
merged_sleep_data = merged_sleep_data.drop('sleepScores', axis=1)
final_df = merged_sleep_data.join(sleep_scores_expanded)
final_df.to_csv('merged_sleep_data.csv', index=False)

pd.set_option('display.max_columns', None)
sleep=pd.read_csv('/content/merged_sleep_data.csv')
sleep = sleep.drop(['sleepWindowConfirmationType', 'unmeasurableSeconds','retro','avgSleepStress','spo2SleepSummary','feedback','insight'], axis=1)
sleep=sleep.sort_values(by='calendarDate')
sleep

sleep['deepSleepSeconds'] = pd.to_timedelta(sleep['deepSleepSeconds'], unit='S')
sleep[['lightSleepSeconds', 'remSleepSeconds','awakeSleepSeconds']]=sleep[['lightSleepSeconds', 'remSleepSeconds','awakeSleepSeconds']].apply(lambda x: pd.to_timedelta(x, unit='S'))
sleep=sleep.rename(columns={"deepSleepSeconds": "deepSleepHours", "lightSleepSeconds": "lightSleepHours", "remSleepSeconds": "remSleepHours","awakeSleepSeconds": "awakeSleepHours"})
sleep['totalSleepHours']= sleep['deepSleepHours'] + sleep['lightSleepHours'] + sleep['remSleepHours'] + sleep['awakeSleepHours']
sleep
