### THIS SCRIPT WILL PRODUCE THE PLOTS AND STATISTICAL RESULTS FROM EXPERIMENT2
### RESULTS WILL BE SAVED IN experiment2/experiment2_results

########################################################################################################################
#PARAMS THAT CAN BE CHANGED:

import os

# Searching fo global variables, if not "Defaults"
try:
   load_results = os.environ['load_results']
except KeyError:
   load_results = "True"                                    # USER CAN MANUALLY CHANGE THIS TO USE RESULTS FROM 31_experiment1.py. IF "True", IT WILL LOAD THE EXPERIMENT1 RESULTS FROM THE ERP RESEARCH. IF "False", IT WILL USE THE RESULTS OF 31_experiment1.py


pollutants = ["NOx",
              "O3",
              "PM2.5"
              ]                                              # LOOP THROUGH THE POLLUTANTS TO CREATE PLOTS FOR EACH POLLUTANT

stations = ["Manchester_Piccadilly",
            "Manchester_Sharston",
            "Salford_Eccles",
            "Salford_Glazebury",
            "Bury_Whitefield"
            ]                                                 # LOOP THROUGH THE STATIONS

#########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression

# Saving color pallet for results (plots)
cmap = plt.colormaps['plasma']



if load_results == "True":
    forecasts = pd.read_csv("saved_results/datasets/forecasts_experiment3.csv")

elif load_results == "False":
    forecasts = pd.read_csv("results/datasets/forecasts_experiment3.csv")


# Converting dates in datetime format
forecasts["ds"] = pd.to_datetime(forecasts["ds"], errors = "raise")

for station in stations:
    for pollutant in pollutants:

        if forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["station"] == station)].shape[0] > 1:

            if pollutant == "NOx":
                co = cmap(80)
            elif pollutant == "PM2.5":
                co = cmap(10)
            elif pollutant == "O3":
                co = cmap(200)

            #Creating directory for saving results
            output_dir = f"experiment3_results/{station}/{pollutant}"
            os.makedirs(output_dir, exist_ok=True)

            print(f"{station} - {pollutant}")
            sns.set_context("paper",font_scale=1.7)
            sns.set_style("white")

            # Plot 1 - Result Prophet

            data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["station"] == station)]

            sns.set_context("paper",font_scale=1.7)
            sns.set_style("white")


            f, ax = plt.subplots(figsize = (20,10))

            sns.lineplot(data = data, x ="ds", y = "yhat_upper", c = co, label = "Uncertainty", alpha = 0.15, zorder = 2)
            sns.lineplot(data = data, x ="ds", y = "yhat_lower", c = co, alpha = 0.15, zorder = 3 )
            plt.fill_between(data =data,x ="ds", y1 = "yhat_upper", y2 = "yhat_lower", color= co, alpha = 0.15, zorder = 1)
            sns.scatterplot(data = data, x ="ds", y = "ALLVAR", c = "black", linewidth = 0,  label = "ALLVAR_measures" )
            sns.lineplot(data = data, x = "ds", y = "yhat", c = co, label="Forecast", linewidth = 4)
            sns.lineplot(data = data, x = "ds", y = "trend", c = "red", label="ProphetTrend", linewidth = 3)

            ax.set(ylabel = f"{pollutant}(μg/m3)")
            ax.set(xlabel = "Date")
            ax.set(title = f"{station} - {pollutant}")


            ax.locator_params(axis='y', nbins=4)

            for spine in ax.spines.values():
                        spine.set_visible(False)

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Prophet_Bias.png', dpi = 300)


            # Plot 2 - Bias

            f, ax = plt.subplots(figsize = (16,10))


            sns.scatterplot(data = data, x ="ds", y = "ALLVAR", c = "black", linewidth = 0,  label = "ALLVAR_measures" )
            sns.lineplot(data = data, x = "ds", y = "yhat", c = co, label="Forecast", linewidth = 4)
            sns.lineplot(data = data, x = "ds", y = "trend", c = "red", label="ProphetTrend", linewidth = 3)


             # Add the X marker at a specific point (example: "01/06/2020", yhat value 50)
            specific_date = pd.to_datetime("25/03/2020", dayfirst=True)
            specific_value = data[(data["ds"] > pd.to_datetime("19/02/2020 00:00:00", dayfirst=True)) & (data["ds"] < pd.to_datetime("19/03/2020 00:00:00", dayfirst=True))].ALLVAR.mean()   #The 19/03 the PM make the announcement of the lockdown

            ax.scatter(specific_date, specific_value, color='fuchsia', s=200, marker='x', zorder=5, label="ALLVAR_Monthly_Mean", linewidth = 3)


            ax.legend()
            ax.set(ylabel = f"{pollutant}(μg/m3)")
            ax.set(xlabel = "Date")
            ax.set(title = f"{station} - {pollutant}")

            ax.locator_params(axis='y', nbins=4)

            for spine in ax.spines.values():
                spine.set_visible(False)

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Prophet_Result.png', dpi = 300)


            print(f"✅Finished with {pollutant}")
        else:
         print("No data for this station and pollutant")
    print(f"✅Finished with {station}")

periods = [
    [pd.to_datetime("01/01/2020 00:00:00", dayfirst=True), pd.to_datetime("10/05/2020 00:00:00", dayfirst=True)],
    [pd.to_datetime("10/05/2020 00:00:00", dayfirst=True), pd.to_datetime("05/11/2020 00:00:00", dayfirst=True)],
    [pd.to_datetime("05/11/2020 00:00:00", dayfirst=True), pd.to_datetime("22/02/2021 00:00:00", dayfirst=True)],
    [pd.to_datetime("22/02/2021 00:00:00", dayfirst=True), pd.to_datetime("01/07/2021 00:00:00", dayfirst=True)]
]


def percentage_change(start_value, end_value):
    return ((end_value - start_value) / start_value) * 100

def slope_per_week_lr(start_date, end_date, df):
    df_period = df.loc[start_date:end_date]

    # Preparing data for linear regression
    X = np.arange(len(df_period)).reshape(-1, 1)
    y = df_period['trend'].values

    model = LinearRegression()
    model.fit(X, y)

    slope_per_hour = model.coef_[0]

    slope_per_week = slope_per_hour * 24 * 7  # 24 hours per day, 7 days per week

    return slope_per_week

results = []

for station in stations:
    for pollutant in pollutants:
        data = forecasts[
            (forecasts["pollutant"] == pollutant) &
            (forecasts["station"] == station)
        ]

        if data.shape[0] > 0:
            df = data[["ds","trend"]]
            complete_index = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="H")
            df = df.set_index("ds").reindex(complete_index)
            df = df.interpolate(method ="linear")

            for period in periods:
                start_date, end_date = period
                start_value = df.loc[start_date]['trend']
                end_value = df.loc[end_date]['trend']

                pct_change = round(percentage_change(start_value, end_value),2)
                period_slope = round(slope_per_week_lr(start_date, end_date, df),2)

                results.append({
                    'Period': f"{start_date.date()} to {end_date.date()}",
                    'initial': start_value,
                    'last': end_value,
                    '% Change': pct_change,
                    'Slope (μg/week)': period_slope,
                    'Station': station,
                    'Pollutant': pollutant
                })

results_df = round(pd.DataFrame(results),2)
results_df.to_csv("experiment3_results/periods_statistics.csv", index=False)
print(results_df)
print("✅✅✅ PLOTS AND METRICS SAVED IN experiment2_results/ DIRECTORY ✅✅✅")
