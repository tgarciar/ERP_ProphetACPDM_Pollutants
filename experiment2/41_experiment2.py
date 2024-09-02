### THIS SCRIPT WILL RUN THE EXPERIMENT 2 AND SAVE THE RESULTS IN experiments/experiment2_results/datasets
### WHEN RUNNING THIS SCRIPT IT TOOK ALMOST 1 HOUR TO RUN. FOR CHECKING HOW IT WORKS IT IS RECOMMENDED TO REDUCE THE NUMBER OF POLLUTANTS BY COMMENTING THEM

#################################################################################
#PARAMS THAT CAN BE CHANGED:
import os

try:
    erp_data = os.environ['ERP_DATA']
except KeyError:
    erp_data = "True"                             # USER CAN MANUALLY CHANGE THIS TO USE PERSONALISED RESULTS. "True" MEANS THAT IT WILL USE THE ERP_WETNOR_TIMESERIES, "False" MEANS IT WILL USE THE RESULTS OF THE PERSONALISED MODEL_TUNING (weather_normalisation/personalised_weather_normalisation_results/WETNOR_timeseries)


stations = ["Manchester_Piccadilly",
            "Manchester_Sharston",
            "Salford_Eccles",
            "Salford_Glazebury",
            "Bury_Whitefield"
            ]
pollutants = ["NOx",
              "PM2.5",
              "O3"
              ]


##################################################################################


import numpy as np
import pandas as pd
import os
import warnings
from prophet import Prophet


# Suppressing FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)



try:
    erp_data = os.environ['ERP_DATA']
except KeyError:
    erp_data = "True"                             # USER CAN MANUALLY CHANGE THIS TO USE PERSONALISED RESULTS. TRUE MEANS THAT IT WILL USE THE ERP_WETNOR_TIMESERIES, FALSE MEANS IT WILL USE THE RESULTS OF THE PERSONALISED MODEL_TUNING (weather_normalisation/personalised_weather_normalisation_results/WETNOR_timeseries)


# Loading the results
if erp_data == "True":
    full_dataset = pd.read_csv("../weather_normalisation/saved_results/WETNOR_timeseries/ERP_WETNOR_combined_ts.csv")
elif erp_data == "False":
    full_dataset = pd.read_csv("../weather_normalisation/personalised_weather_normalisation_results/WETNOR_timeseries/personalised_WETNOR_combined_ts.csv")

# Setting date as DateTime
full_dataset["date"] = pd.to_datetime(full_dataset["date"], errors="raise")


# Filtering ALLVAR time series
full_dataset = full_dataset[["date", "ALLVAR", "station", "pollutant"]]


# Setting dataframes that will help to save results:
trends_dataframe = pd.DataFrame(columns = ["trend","station", "pollutant", "type"])
trends_dataframe.index.name = 'date'

changepoint_magnitudes_change_dataframe = pd.DataFrame(columns = ["ds","rate_of_change","type","station","pollutant"])

forecasts_dataframe = pd.DataFrame(columns = ["ds","trend","yhat_lower", "yhat","yhat_upper", "station","pollutant","type"])
# Looping to trained the model on different pollutants
for station in stations:

    for pollutant in pollutants:
        temp_dataset = full_dataset[(full_dataset["station"] == station) & (full_dataset["pollutant"] == pollutant)]

        model = []
        print(f"Starting with... {station} - {pollutant} - ALLVAR")
        print("")

        # Setting model input
        working_dataset = temp_dataset[["date","ALLVAR"]].rename(columns={
        'date': 'ds',
        "ALLVAR": 'y'
        })
         # Checking that Nulls are less than 80% of the dataset (Same as in weather normalisation)
        if len(working_dataset.dropna()) > 0.8 * len(working_dataset):

            # Creating the possible changepoints: Weekly --> There are different dates of start and end, that is why is inside the loop.
            min_date = temp_dataset["date"].min()
            max_date = temp_dataset["date"].max()
            start_date = f'{min_date}'
            end_date = f'{max_date}'

            date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')

            weekly_changepoints = pd.DataFrame(date_range, columns=['changepoint'])
            weekly_changepoints['changepoint'] = weekly_changepoints['changepoint'].dt.strftime('%Y-%m-%d %H:%M:%S')


            ## Running Prophet
            model = Prophet(interval_width=1, changepoint_range=1, changepoints = weekly_changepoints["changepoint"])

            model.fit(working_dataset)

            # Creating Forecasting dataframe: NO FORECASTING (0 Hours)
            model_future_dates = model.make_future_dataframe(periods=0, freq="h")

            # Forecasting to get the results
            forecast_model = model.predict(model_future_dates)

            ## Creating rate of changes on Prophet Trend(magnitude of changes) using changepoints

            # Extracting trend from forecast and changepoints:
            trend = forecast_model[["ds", "trend"]].set_index("ds")
            changepoints = model.changepoints

            # Interpolating GAPS in trends --> Avoid overestimation of changes.
            min_date = forecast_model["ds"].min()
            max_date = forecast_model["ds"].max()
            start_date = f'{min_date}'
            end_date = f'{max_date}'

            #Creating weekly frequency to grab all the weeks
            date_range = pd.date_range(start=start_date, end=end_date, freq='h')
            #All hours
            hourly_dates = pd.DataFrame(date_range, columns=['date'])
            hourly_dates['date'] = pd.to_datetime(hourly_dates['date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
            hourly_dates.set_index("date", inplace = True)

            #Merging and Interpolating all the trends gaps
            trend_interpolate = hourly_dates.merge(right=trend, how="left", left_index=True, right_index=True)
            trend_interpolate = trend_interpolate.interpolate(method="linear")

            #Filtering only during changepoints
            rate_of_change = trend_interpolate.loc[trend_interpolate.index.isin(changepoints), 'trend'].diff().reset_index(drop = True)

            # Creating a DataFrame with the reuslts of changepoints

            changepoint_magnitudes_change = pd.DataFrame({
                'ds': changepoints,
                'rate_of_change': rate_of_change,
                'type': "ALLVAR",
                'station': station,
                'pollutant': pollutant
            })

            #Saving results:

            # Trend
            trend_interpolate["station"] = station
            trend_interpolate["pollutant"] = pollutant
            trend_interpolate["type"] = "ALLVAR"
            trends_dataframe = pd.concat([trends_dataframe,trend_interpolate])



            # Changepoints magnitudes
            changepoint_magnitudes_change_dataframe = pd.concat([changepoint_magnitudes_change_dataframe,changepoint_magnitudes_change])

            # Forecast
            forecast = forecast_model[["ds","trend","yhat_lower", "yhat","yhat_upper"]]
            forecast["station"] = station
            forecast["pollutant"] = pollutant
            forecast["type"] = "ALLVAR"
            forecast["ALLVAR"] = working_dataset.reset_index()["y"]


            forecasts_dataframe = pd.concat([forecasts_dataframe,forecast])

        else:
            print(f"No data for {station} - {pollutant} - ALLVAR")
            print("")

output = f"results/datasets/"
os.makedirs(output, exist_ok=True)
forecasts_dataframe.to_csv(f'{output}/forecasts_experiment2.csv', index = False)
trends_dataframe.to_csv(f'{output}/trends_experiment2.csv', index = True)
changepoint_magnitudes_change_dataframe.to_csv(f'{output}/changepoint_magnitudes_experiment2.csv', index = False)
