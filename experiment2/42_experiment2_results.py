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
              ]                          # LOOP THROUGH THE POLLUTANTS TO CREATE PLOTS FOR EACH POLLUTANT

stations = ["Manchester_Piccadilly",
            "Manchester_Sharston",
            "Salford_Eccles",
            "Salford_Glazebury",
            "Bury_Whitefield"
            ]  # LOOP THROUGH THE STATIONS

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
    changepoint_magnitudes = pd.read_csv("saved_results/datasets/changepoint_magnitudes_experiment2.csv")
    forecasts = pd.read_csv("saved_results/datasets/forecasts_experiment2.csv")

elif load_results == "False":
    changepoint_magnitudes = pd.read_csv("results/datasets/changepoint_magnitudes_experiment2.csv")
    forecasts = pd.read_csv("results/datasets/forecasts_experiment2.csv")


# Converting dates in datetime format
changepoint_magnitudes["ds"] = pd.to_datetime(changepoint_magnitudes["ds"], errors = "raise")
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
            output_dir = f"experiment2_results/{station}/{pollutant}"
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

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Prophet_Result.png', dpi = 300)

            # Plot 2 - Changepoints
            f, ax = plt.subplots(figsize = (14,10))

            changepoint_temp = changepoint_magnitudes[(changepoint_magnitudes["pollutant"] == pollutant) & (changepoint_magnitudes["station"] == station)]

            changepoint_temp["ds"] = pd.to_datetime(changepoint_temp["ds"])

            # Plot the magnitude of the rate change at each changepoint

            plt.axhline(y= 0, color='black', linestyle='-', linewidth = 1,label='Rate change of 0')

            plt.plot(changepoint_temp['ds'], changepoint_temp['rate_of_change'], marker = 'o',  linewidth = 3, color = co, markersize=10, label = "WeeklyChangepoint")
            plt.xlabel('Date')
            plt.ylabel(f'Rate of Change {pollutant}(μg/m2)')

            plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 , label='Lockdowns')
            plt.axvline(x=pd.to_datetime("10/05/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("02/12/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("22/02/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("15/06/2020 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 ,label='Non-essential shops reopenning')
            plt.axvline(x=pd.to_datetime("03/08/2020 09:00:00", dayfirst = True), color = "darkcyan", linestyle='--',linewidth = 2 ,label='Eat out to Help Out')
            plt.axvline(x=pd.to_datetime("12/04/2021 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 )


            plt.axvspan(pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), pd.to_datetime("10/05/2020 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), pd.to_datetime("02/12/2020 09:00:00", dayfirst = True),color =cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), pd.to_datetime("22/02/2021 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)


            plt.xlim(pd.to_datetime("01/01/2020 00:00:00", dayfirst = True),pd.to_datetime("01/07/2021 00:00:00", dayfirst = True))

            plt.title(f'{station} - {pollutant}:Rate of Change at Weekly Changepoints')

            ax.legend()

            plt.savefig(f'{output_dir}/{station}_{pollutant}_weekly_rate_change.png', dpi = 300)

            # Plot 3 - CUMSUM

            cp_cumsum = changepoint_temp[(changepoint_temp["ds"] > "01/01/2020") & (changepoint_temp["ds"] < "08/01/2021")]

            cp_cumsum["cumsum"] = cp_cumsum["rate_of_change"].cumsum()

            # Plot the CUMSUM of the rate change at each changepoint
            f, ax = plt.subplots(figsize = (14,10))

            plt.axhline(y= 0, color='black', linestyle='-', linewidth = 0.5,label='Rate change of 0')

            plt.plot(cp_cumsum['ds'], cp_cumsum['cumsum'], marker = 'o',  linewidth = 3 , color = co, markersize=10)
            plt.xlabel('Date')
            plt.ylabel(f'CUMSUM Rate of Change {pollutant}(μg/m2)')

            plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 , label='Lockdowns')
            plt.axvline(x=pd.to_datetime("10/05/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("02/12/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("22/02/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("15/06/2020 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 ,label='Non-essential shops reopenning')
            plt.axvline(x=pd.to_datetime("03/08/2020 09:00:00", dayfirst = True), color = "darkcyan", linestyle='--',linewidth = 2 ,label='Eat out to Help Out')
            plt.axvline(x=pd.to_datetime("12/04/2021 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 )


            plt.axvspan(pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), pd.to_datetime("10/05/2020 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), pd.to_datetime("02/12/2020 09:00:00", dayfirst = True),color =cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), pd.to_datetime("22/02/2021 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)


            plt.xlim(pd.to_datetime("01/01/2020 00:00:00", dayfirst = True),pd.to_datetime("01/08/2021 00:00:00", dayfirst = True))

            plt.title(f'{station} - {pollutant}: CUMSUM Rate of Change at Weekly Changepoints')
            for spine in ax.spines.values():
                        spine.set_visible(False)


            plt.savefig(f'{output_dir}/{station}_{pollutant}_CUMSUM_weekly_change.png', dpi = 300)

            # Plot 4 - Period 1

            forecast_pollutant = forecasts[
            (forecasts["ds"] > pd.to_datetime("01/01/2020 00:00:00", dayfirst=True)) &
            (forecasts["ds"] < pd.to_datetime("10/05/2020 00:00:00", dayfirst=True)) &
            (forecasts["pollutant"] == pollutant) &
            (forecasts["station"] == station)
        ]


            fig3, ax = plt.subplots(figsize = (10,10))

            plt.plot(forecast_pollutant['ds'], forecast_pollutant['trend'],  linewidth = 3, color = co, markersize=10, label = "ProphetTrend")
            plt.xlabel('Date')
            plt.ylabel(f'Trend {pollutant} (μg/m2)')

            plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 , label='Lockdowns')
            plt.axvline(x=pd.to_datetime("10/05/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("02/12/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("22/02/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("15/06/2020 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 ,label='Non-essential shops reopenning')
            plt.axvline(x=pd.to_datetime("03/08/2020 09:00:00", dayfirst = True), color = "darkcyan", linestyle='--',linewidth = 2 ,label='Eat out to Help Out')
            plt.axvline(x=pd.to_datetime("12/04/2021 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 )


            plt.axvspan(pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), pd.to_datetime("10/05/2020 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), pd.to_datetime("02/12/2020 09:00:00", dayfirst = True),color =cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), pd.to_datetime("22/02/2021 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)

            plt.xlim(pd.to_datetime("01/01/2020 00:00:00", dayfirst = True),pd.to_datetime("10/05/2020 00:00:00", dayfirst = True))

            plt.title('Period1')

            ax.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
            ax.legend()
            plt.xticks([pd.to_datetime("01/01/2020", dayfirst=True), pd.to_datetime("10/05/2020", dayfirst=True)])

            for spine in ax.spines.values():
                        spine.set_visible(False)

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Period1.png', dpi = 300)

            # Plot 5 - Period 2
            forecast_pollutant = forecasts[
            (forecasts["ds"] > pd.to_datetime("10/05/2020 00:00:00", dayfirst=True)) &
            (forecasts["ds"] < pd.to_datetime("05/11/2020 00:00:00", dayfirst=True)) &
            (forecasts["pollutant"] == pollutant) &
            (forecasts["station"] == station)
            ]

            fig4, ax = plt.subplots(figsize = (10,10))

            plt.plot(forecast_pollutant['ds'], forecast_pollutant['trend'],  linewidth = 3, color = co, markersize=10, label = "ProphetTrend")
            plt.xlabel('Date')
            plt.ylabel(f'Trend {pollutant} (μg/m2)')

            plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 , label='Lockdowns')
            plt.axvline(x=pd.to_datetime("10/05/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("02/12/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("22/02/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("15/06/2020 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 ,label='Non-essential shops reopenning')
            plt.axvline(x=pd.to_datetime("03/08/2020 09:00:00", dayfirst = True), color = "darkcyan", linestyle='--',linewidth = 2 ,label='Eat out to Help Out')
            plt.axvline(x=pd.to_datetime("12/04/2021 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 )


            plt.axvspan(pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), pd.to_datetime("10/05/2020 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), pd.to_datetime("02/12/2020 09:00:00", dayfirst = True),color =cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), pd.to_datetime("22/02/2021 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)

            plt.xlim(pd.to_datetime("10/05/2020 00:00:00", dayfirst = True),pd.to_datetime("05/11/2020 00:00:00", dayfirst = True))

            plt.title('Period2')

            ax.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
            plt.xticks([pd.to_datetime("10/05/2020", dayfirst=True), pd.to_datetime("05/11/2020", dayfirst=True)])

            for spine in ax.spines.values():
                        spine.set_visible(False)

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Period2.png', dpi = 300)

            # Plot 6 - Period 3
            forecast_pollutant = forecasts[
            (forecasts["ds"] > pd.to_datetime("05/11/2020 00:00:00", dayfirst=True)) &
            (forecasts["ds"] < pd.to_datetime("22/02/2021 00:00:00", dayfirst=True)) &
            (forecasts["pollutant"] == pollutant) &
            (forecasts["station"] == station)
            ]

            fig5, ax = plt.subplots(figsize = (10,10))

            plt.plot(forecast_pollutant['ds'], forecast_pollutant['trend'],  linewidth = 3, color = co, markersize=10, label = "ProphetTrend")
            plt.xlabel('Date')
            plt.ylabel(f'Trend {pollutant} (μg/m2)')

            plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 , label='Lockdowns')
            plt.axvline(x=pd.to_datetime("10/05/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("02/12/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("22/02/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("15/06/2020 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 ,label='Non-essential shops reopenning')
            plt.axvline(x=pd.to_datetime("03/08/2020 09:00:00", dayfirst = True), color = "darkcyan", linestyle='--',linewidth = 2 ,label='Eat out to Help Out')
            plt.axvline(x=pd.to_datetime("12/04/2021 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 )


            plt.axvspan(pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), pd.to_datetime("10/05/2020 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), pd.to_datetime("02/12/2020 09:00:00", dayfirst = True),color =cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), pd.to_datetime("22/02/2021 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)

            plt.xlim(pd.to_datetime("05/11/2020 00:00:00", dayfirst = True),pd.to_datetime("22/02/2021 00:00:00", dayfirst = True))

            plt.title('Period3')

            ax.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
            plt.xticks([pd.to_datetime("05/11/2020", dayfirst=True), pd.to_datetime("22/02/2021", dayfirst=True)])

            for spine in ax.spines.values():
                        spine.set_visible(False)

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Period3.png', dpi = 300)

            forecast_pollutant = forecasts[
            (forecasts["ds"] > pd.to_datetime("22/02/2021 00:00:00", dayfirst=True)) &
            (forecasts["ds"] < pd.to_datetime("01/07/2021 00:00:00", dayfirst=True)) &
            (forecasts["pollutant"] == pollutant) &
            (forecasts["station"] == station)
            ]

            fig6, ax = plt.subplots(figsize = (10,10))

            plt.plot(forecast_pollutant['ds'], forecast_pollutant['trend'],  linewidth = 3, color = co, markersize=10, label = "ProphetTrend")
            plt.xlabel('Date')
            plt.ylabel(f'Trend {pollutant} (μg/m2)')

            plt.axvline(x=pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 , label='Lockdowns')
            plt.axvline(x=pd.to_datetime("10/05/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("02/12/2020 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("22/02/2021 09:00:00", dayfirst = True), color = cmap(10), linestyle='--',linewidth = 1.5 )
            plt.axvline(x=pd.to_datetime("15/06/2020 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 ,label='Non-essential shops reopenning')
            plt.axvline(x=pd.to_datetime("03/08/2020 09:00:00", dayfirst = True), color = "darkcyan", linestyle='--',linewidth = 2 ,label='Eat out to Help Out')
            plt.axvline(x=pd.to_datetime("12/04/2021 09:00:00", dayfirst = True), color = "darkorange", linestyle='--',linewidth = 2 )


            plt.axvspan(pd.to_datetime("26/03/2020 09:00:00", dayfirst = True), pd.to_datetime("10/05/2020 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("05/11/2020 09:00:00", dayfirst = True), pd.to_datetime("02/12/2020 09:00:00", dayfirst = True),color =cmap(10), alpha=0.1)
            plt.axvspan(pd.to_datetime("06/01/2021 09:00:00", dayfirst = True), pd.to_datetime("22/02/2021 09:00:00", dayfirst = True),color = cmap(10), alpha=0.1)

            plt.xlim(pd.to_datetime("22/02/2021 00:00:00", dayfirst = True),pd.to_datetime("01/07/2021 00:00:00", dayfirst = True))

            plt.title('Period4')

            ax.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
            plt.xticks([pd.to_datetime("22/02/2021", dayfirst=True), pd.to_datetime("01/07/2021", dayfirst=True)])

            for spine in ax.spines.values():
                        spine.set_visible(False)

            plt.savefig(f'{output_dir}/{station}_{pollutant}_Period4.png', dpi = 300)
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
results_df.to_csv("experiment2_results/periods_statistics.csv", index=False)
print(results_df)
print("✅✅✅ PLOTS AND METRICS SAVED IN experiment2_results/ DIRECTORY ✅✅✅")
