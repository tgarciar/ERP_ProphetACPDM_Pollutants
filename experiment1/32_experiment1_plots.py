### THIS SCRIPT WILL PRODUCE THE PLOTS FROM EXPERIMENT1
### RESULTS WILL BE SAVED IN experiment1/experiment1_plots
########################################################################################################################
#PARAMS THAT CAN BE CHANGED:

import os

# Searching fo global variables, if not "Defaults"
try:
   load_results = os.environ['load_results']
except KeyError:
   load_results = "True"                                    # USER CAN MANUALLY CHANGE THIS TO USE RESULTS FROM 31_experiment1.py. IF "True", IT WILL LOAD THE EXPERIMENT1 RESULTS FROM THE ERP RESEARCH. IF "False", IT WILL USE THE RESULTS OF 31_experiment1.py


pollutants = ["NOx", "O3","PM2.5"]                          # LOOP THROUGH THE POLLUTANTS TO CREATE PLOTS FOR EACH POLLUTANT


#########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# Saving color pallet for results (plots)
cmap = plt.colormaps['plasma']



if load_results == "True":
    changepoint_magnitudes = pd.read_csv("saved_results/datasets/changepoint_magnitudes_experiment1.csv")
    forecasts = pd.read_csv("saved_results/datasets/forecasts_experiment1.csv")
    full_dataset = pd.read_csv("../weather_normalisation/saved_results/WETNOR_timeseries/ERP_WETNOR_combined_ts.csv")

elif load_results == "False":
    changepoint_magnitudes = pd.read_csv("results/datasets/changepoint_magnitudes_experiment1.csv")
    forecasts = pd.read_csv("results/datasets/forecasts_experiment1.csv")
    full_dataset = pd.read_csv("../weather_normalisation/personalised_weather_normalisation_results/WETNOR_timeseries/personalised_WETNOR_combined_ts.csv")


# Converting dates in datetime format
full_dataset["date"] = pd.to_datetime(full_dataset["date"], errors = "raise")
changepoint_magnitudes["ds"] = pd.to_datetime(changepoint_magnitudes["ds"], errors = "raise")
forecasts["ds"] = pd.to_datetime(forecasts["ds"], errors = "raise")


for pollutant in pollutants:
    # Filtering the data for the station and pollutant
    data = full_dataset[(full_dataset["pollutant"] == pollutant) & (full_dataset["station"] == "Manchester_Piccadilly")]


    #Creating directory for saving results
    output_dir = f"experiment1_plots/{pollutant}"
    os.makedirs(output_dir, exist_ok=True)

    # Setting context for the plots
    sns.set_context("paper",font_scale=1.7)
    sns.set_style("white")
    # Plot 1 -
    f, ax = plt.subplots(figsize = (18,10))

    sns.lineplot(data = data, x = "date", y = "measure", c = cmap(10), label="Observation", linewidth = 2)
    sns.lineplot(data = data, x = "date", y = "METVAR", c = cmap(100), label="METVAR_RandomSearch", linewidth = 2)
    sns.lineplot(data = data, x = "date", y = "ALLVAR", c = cmap(200), label="ALLVAR_RandomSearch", linewidth = 2)

    ax.set(ylabel = f"{pollutant}(μg/m3)")
    ax.set(xlabel = "Date")


    inset_ax = f.add_axes([0.5, 0.5, 0.395, 0.37], projection = "rectilinear")
    zoom_start, zoom_end = '2020-02-01', '2020-03-01'

    zoomed_data = data[(data['date'] >= zoom_start) & (data['date'] <zoom_end)]
    sns.lineplot(x = "date", y = "measure", c = cmap(10), data=zoomed_data, ax=inset_ax, linewidth = 2)
    sns.lineplot(x = "date", y = "METVAR", c = cmap(100), data=zoomed_data, ax=inset_ax, linewidth = 2)
    sns.lineplot(x = "date", y = "ALLVAR", c = cmap(200), data=zoomed_data, ax=inset_ax, linewidth = 2)


    inset_ax.set(xlabel = "")
    inset_ax.set(ylabel = "")
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.tick_params(left=False, bottom=False)
    inset_ax.spines['top'].set_color('mediumturquoise')
    inset_ax.spines['top'].set_linewidth(2)
    inset_ax.spines['right'].set_color('mediumturquoise')
    inset_ax.spines['right'].set_linewidth(2)
    inset_ax.spines['bottom'].set_color('mediumturquoise')
    inset_ax.spines['bottom'].set_linewidth(2)
    inset_ax.spines['left'].set_color('mediumturquoise')
    inset_ax.spines['left'].set_linewidth(2)

    rect = plt.Rectangle((pd.to_datetime(zoom_start), zoomed_data['measure'].min()),
                        pd.to_datetime(zoom_end) - pd.to_datetime(zoom_start),
                        zoomed_data['measure'].max() - zoomed_data['measure'].min(),
                        linewidth=2, edgecolor='mediumturquoise', facecolor='none', zorder = 5, linestyle='--')
    ax.add_patch(rect)


    line1 = plt.Line2D([pd.to_datetime("2020-02-01"),pd.to_datetime("2019-06-01")], [340,450], color='mediumturquoise', linestyle='--', linewidth = 2)
    line2 = plt.Line2D([pd.to_datetime("2020-03-01"),pd.to_datetime("2022-12-30")], [340,450], color='mediumturquoise', linestyle='--', linewidth = 2)

    ax.add_line(line1)
    ax.add_line(line2)

    ax.locator_params(axis='y', nbins=4)

    ax.legend(title = "Method", loc = "upper left")
    ax.set_title(f"{pollutant} in Manchester Piccadilly - Comparison of Observations WETNOR time series")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{pollutant}_comparison.png")
    plt.close()

    # Plot 2 -  Trend by WETNOR {pollutant  }
    f, ax = plt.subplots(figsize = (9,8))

    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "OBSERVATION")], x = "ds", y = "trend",  c = cmap(10), label="Observation", linewidth = 3)
    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "METVAR_normet")], x = "ds", y = "trend",c = cmap(120) ,label = "METVAR_Normet", linewidth = 3)
    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "METVAR")], x = "ds", y = "trend", c = cmap(70), label="METVAR_RandomSearch", linewidth = 3)
    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "ALLVAR_normet")], x = "ds", y = "trend", c = cmap(220),label = "ALLVAR_Normet", linewidth = 3)
    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "ALLVAR")], x = "ds", y = "trend", c = cmap(180), label="ALLVAR_RandomSearch", linewidth = 3)

    plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = "forestgreen", linestyle='--',linewidth = 3 ,label='First Lockdown')

    ax.set(ylabel = f"{pollutant}(μg/m3)")
    ax.set(xlabel = "Date")
    ax.set(title= f"Manchester_Piccadilly: {pollutant} Trends by WETNOR time series")
    ax.locator_params(axis='y', nbins=4)


    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{pollutant}_trends.png")

    # Plot 3 -  Trends for ALLVAR and METVAR

    f, ax = plt.subplots(figsize = (12,8))

    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "ALLVAR_normet")], x = "ds", y = "trend", label = "ALLVAR_normet trend", c = cmap(220), linewidth = 3)
    sns.lineplot(data = forecasts[(forecasts["pollutant"] == pollutant) & (forecasts["type"] == "ALLVAR")], x = "ds", y = "trend", label = "ALLVAR_RandomSearch trend",c = cmap(180), linewidth = 3)

    plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = "forestgreen", linestyle='--',linewidth = 3 ,label='First Lockdown')

    ax.set(ylabel = f"{pollutant}(μg/m3)")
    ax.set(xlabel = "Date")
    ax.set(title= f"Manchester_Piccadilly: {pollutant} Trend for ALLVAR and METVAR")
    ax.locator_params(axis='y', nbins=1)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))  # Show one date every 3 months
    ax.legend()

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{pollutant}_trends_ALLVAR.png")

    sns.set_style("white")

    #Plot 4 - Rate of change for ALLVAR and METVAR
    f, ax = plt.subplots(figsize = (12,8))

    sns.lineplot(data = changepoint_magnitudes[(changepoint_magnitudes["pollutant"] == pollutant) & (changepoint_magnitudes["type"] == "ALLVAR_normet")], x = "ds", y = "rate_of_change",marker = 'o',markersize=8, c = cmap(220),label = "Weekly ALLVAR_Normet trend lag", linewidth = 3)
    sns.lineplot(data = changepoint_magnitudes[(changepoint_magnitudes["pollutant"] == pollutant) & (changepoint_magnitudes["type"] == "ALLVAR")], x = "ds", y = "rate_of_change",marker = 'o',markersize=8, c = cmap(180), label="Weekly ALLVAR_RandomSearch trend lag", linewidth = 3)

    plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = "forestgreen", linestyle='--',linewidth = 3 ,label='First Lockdown')
    plt.axhline(y= 0, color='black', linestyle='-', linewidth = 1)

    ax.set(ylabel = f"Magnitude of change:{pollutant}(μg/m3)")
    ax.set(xlabel = "Date")
    ax.set(title= f"Manchester_Piccadilly: Magnitudes of rate of change for {pollutant}")
    ax.locator_params(axis='y', nbins=4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))  # Show one date every 3 months

    ax.set(xlim = (pd.to_datetime("01/01/2019 00:00:00", dayfirst = True),pd.to_datetime("01/07/2020 00:00:00", dayfirst = True)))

    ax.legend()
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{pollutant}_rate_of_change.png")




print("✅✅✅ PLOTS SAVED IN experiment1_plots/ DIRECTORY ✅✅✅")
