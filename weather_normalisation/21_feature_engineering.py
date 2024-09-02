### THIS SCRIPT WILL MERGE THE AURN DATA WITH THE ERA5 REANALYSIS DATA. ADDITIONALLY, IT WILL
### CREATE THE TEMPORAL FEATURES, CYCLICAL FEATURES (SIN AND COS) AND THE WIND CONTINOUS FEATURES.
### THE FINAL DATASET WILL BE SAVED INSIDE THE weather_nomalisation/datasets FOLDER

import numpy as np
import pandas as pd
import datetime as dt
import time
import os

# Loading the AURN dataset
combined_df = pd.read_csv("../EDA_results/datasets/combined_AURN_data.csv")
# Converting date into datetime. Setting date as index
combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y-%m-%d %H:%M:%S", errors = "raise")
combined_df.set_index("date", inplace=True)

# Loading the ERA5 reanalysis dataset
man_era5 = pd.read_csv("../data_retrieval/datasets/ERA5_reanalysis_data/Man_ECMWF_reanalysis_data.csv")
man_era5['date'] = pd.to_datetime(man_era5['time.1'], errors='raise')
man_era5.sort_values(by = "date", inplace=True)

# Dropping repeated columns not used on this report
man_era5.drop(columns = ["time","time.1","blh","blh2" ], inplace = True)


# Merging the AURN and ERA5 datasets
data_ready_wetnor = combined_df.merge(right = man_era5, how = "inner", right_on= "date", left_on= "date")
data_ready_wetnor.set_index("date", inplace=True)

# Double  checking that there are no rows missing after the merge. Using 31/12/2022 as the last date in ERA5 data.
assert combined_df.sort_index().loc[:pd.to_datetime("31/12/2022 23:00:00 ", dayfirst=True)].shape[0] == data_ready_wetnor.shape[0], "The number of rows in the merged dataset is different from the original dataset"

# Feature Engineering
# Adding Temporal data:
data_ready_wetnor["hour"] = data_ready_wetnor.index.hour
data_ready_wetnor["weekday"] = data_ready_wetnor.index.weekday
data_ready_wetnor["month"] = data_ready_wetnor.index.month
data_ready_wetnor['unix'] = (data_ready_wetnor.index - dt.datetime(1970, 1, 1)).total_seconds()
data_ready_wetnor['day_of_year'] = data_ready_wetnor.index.dayofyear

# Creating Cyclical Data - Temporal Features

# Hours Cyclical
data_ready_wetnor['hour_sin'] = round(np.sin(2 * np.pi * data_ready_wetnor['hour'] / 24),5)
data_ready_wetnor['hour_cos'] = round(np.cos(2 * np.pi * data_ready_wetnor['hour'] / 24),5)

# Monthly Cyclical
data_ready_wetnor['month_sin'] = round(np.sin(2 * np.pi * (data_ready_wetnor['month'] - 1)/ 12),5)
data_ready_wetnor['month_cos'] = round(np.cos(2 * np.pi * (data_ready_wetnor['month'] - 1)/ 12),5)

# Daily Cyclical
data_ready_wetnor['day_sin'] = round(np.sin(2 * np.pi * data_ready_wetnor['day_of_year'] / 365),5)
data_ready_wetnor['day_cos'] = round(np.cos(2 * np.pi * data_ready_wetnor['day_of_year'] / 365),5)

# Wind Continous
# Calculating the u (x-component) and v (y-component) of the wind
data_ready_wetnor['wind_u'] = data_ready_wetnor['ws'] * np.cos(np.deg2rad(data_ready_wetnor['wd']))
data_ready_wetnor['wind_v'] = data_ready_wetnor['ws'] * np.sin(np.deg2rad(data_ready_wetnor['wd']))

# Cleaning repeated Features/Correlated data
columns_to_remove = ['u10', 'v10', 't2m', #Repeated data. Preference over AURN site data (more specific to the station)
                    'hour', 'month', 'day_of_year', #  Collinearity with temporals cyclical
                    'wd','ws', #  Collinearity with Wind U, Wind V
]

data_ready_wetnor.drop(columns = columns_to_remove, inplace = True)

output_dir = f"datasets/"
os.makedirs(output_dir, exist_ok=True)



data_ready_wetnor.to_csv(f"{output_dir}/data_for_wetnor.csv")

print(f" ✅✅✅ DATA PREPARATION PROCESS FINISHED!")
