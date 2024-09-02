### IMPORTANT: THIS SCRIPT WILL PLOT A POLLUTION WIND ROSE THAT NEEDS THE USER TO HAVE R AND
### THE R LIBRARY OPENAIR IN THEIR SYSTEM. IF NOT, SET pollution_windrose = False ON LINE 98

### THIS SCRIPT RUNS THE EDA FOR EVERY STATION GIVING AS A RESULT THE IMAGES SHARED IN THE REPORT.
### AFTER RUNNING THIS SCRIPT, THE RESULTS WILL BE SAVED IN THE EDA_results FOLDER.
### THERE IS MORE DETAIL, IF NEEDED, OF THE FULL EDA ON THEEDA JUPYTER NOTEBOOKS ON notebooks/EDA

#################################################################################################
# PARAMS THAT CAN BE CHANGED:


# Setting the windrose to True, the EDA will plot the windrose for pollutants on the stations.
# This uses the R library openair. If crashing, set to False:

pollution_windrose = True

#################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


# Saving color pallet for results (plots)
cmap = plt.colormaps['plasma']

# AURN stations to analyse and their files names
stations = {
    "Manchester_Piccadilly": "man_picc_AURN.csv",
    "Manchester_Sharston": "man_shar_AURN.csv",
    "Salford_Eccles": "sal_eccl_AURN.csv",
    "Salford_Glazebury": "sal_glaz_AURN.csv",
    "Bury_Whitefield": "bury_whit_AURN.csv"
}

pollutants = ["NOx", "PM2.5", "O3"]

# Empty dictionary to save the loaded dataframes
dataframes = {}

# Looping through each station, loading and processing the data
for station, file in stations.items():

    df = pd.read_csv(f"data_retrieval/datasets/AURN_data/{file}")

    # Filtering columns not important for this ERP (More detail in the EDA Jupyter Notebooks)
    df = df[df.columns.intersection(pollutants + ["date", "wd", "ws", "temp"])]

    # Adding the station name as a column
    df["station"] = station

    # Creating datetime column and setting it as the index
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df.sort_values(by="date", inplace=True)
    df.set_index("date", drop=True, inplace=True)

    # Storing the processed DataFrames in the dictionary
    dataframes[station] = df

# Concatenating all DataFrames into one
combined_df = pd.concat(dataframes.values())


# Creating profiling data for each station:
combined_df["hour"] = combined_df.index.hour
combined_df["weekday"] =combined_df.index.weekday
combined_df["week"] = combined_df.index.isocalendar().week
combined_df["dayname"] = combined_df.index.day_name()
combined_df["month"] = combined_df.index.month_name()
combined_df["year"] = combined_df.index.year
combined_df["isweekday"] = combined_df['weekday'].apply(lambda x: "Weekend" if x in [5, 6] else "Weekday")

# Adding a season column
def get_season(date):
    if date.month in [12, 1, 2]:
        return 'Winter'
    elif date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    elif date.month in [9, 10, 11]:
        return 'Autumn'
combined_df['season'] = combined_df.index.map(get_season)

# Changing month and dayname to categorical (This will be useful for the plots)
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
combined_df["dayname"] = pd.Categorical(combined_df["dayname"].str[:3], categories=day_order, ordered=True)
month_order = ["Jan", "Apr", "Jul", "Oct"]
combined_df["month"] = pd.Categorical(combined_df["month"].str[:3], categories=month_order, ordered=True)


# Stations to analyse (This part of the code will create the different plots resulting from the EDA Analyis)
# The user can comment/uncomment the station to analyse

stations_EDA = ["Manchester_Piccadilly",
                "Manchester_Sharston",
                "Salford_Eccles",
                "Salford_Glazebury",
                "Bury_Whitefield"
            ]


for station in stations_EDA:
    output_dir = f"EDA_results/{station}"
    os.makedirs(output_dir, exist_ok=True)

    # Filtering the data for the station
    df = combined_df[combined_df["station"] == station]

    # For looping to throw the stations with NAN values in pollutants. This is because
    # not all stations have pollutants available but when we concatenate
    for pollutant in pollutants:
        if df[pollutant].isna().sum() == 0:
            continue
        else :
            df = df.drop(columns=pollutant)


    # Creating dataframe with daily resampling, mean strategy on pollutants data
    df_day = df.copy()
    df_day = df_day[df_day.columns.intersection(pollutants)].resample("D").mean()

    #Setting the style of the plots
    sns.set_context("paper",font_scale=1.7)
    sns.set_style("white")

    # Plotting the daily average of pollutants

    # First plot: Daily average of pollutants
    f, ax = plt.subplots(figsize=(25, 10))

    sns.lineplot(data = df_day, x=df_day.index, c = cmap(90), y = "NOx", label = "NOx", linewidth  =2 )
    # Checking that the pollutant is in the columns before adding it to the plot
    if "PM2.5" in df_day.columns:
        sns.lineplot(data = df_day, x=df_day.index, c = cmap(10), y = "PM2.5", label = "PM2.5", linewidth  =2)
    if "O3" in df_day.columns:
        sns.lineplot(data = df_day, x=df_day.index, c = cmap(200), y = "O3", label = "O3", linewidth  =2)

    ax.set(ylabel = "μg/m3")
    ax.set(xlabel = "Date")
    ax.set(title = f"{station}:Daily Concentration")

    #Showing x axis every 2 years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.locator_params(axis='y', nbins=4)
    ax.legend(title = "Pollutant")

    # Taking spines off
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{station}_Daily_Concentration.png")
    plt.close()


    # Second plot: Daily profiling of pollutants - in a week

    f, ax = plt.subplots(figsize = (16,8))

    sns.lineplot(data = df, x="dayname", c = cmap(90), y = "NOx", label = "NOx", linewidth  =2 )
    # Checking that the pollutant is in the columns before adding it to the plot
    if "PM2.5" in df_day.columns:
        sns.lineplot(data = df, x="dayname", c = cmap(10), y = "PM2.5", label = "PM2.5", linewidth  =2)
    if "O3" in df_day.columns:
        sns.lineplot(data = df, x="dayname", c = cmap(200), y = "O3", label = "O3", linewidth  =2)

    ax.set(title= "Concentration by Day")
    ax.legend(title = "Pollutant")
    ax.set(xlabel = "Day of the Week")
    ax.set(ylabel = "μg/m3")
    ax.locator_params(axis='y', nbins=4)

     # Taking spines off
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{station}_DayOfTheWeek_Profiling.png")
    plt.close()

    # Third plot: Monthly profiling of pollutants
    f, ax = plt.subplots(figsize = (16,8))

    sns.lineplot(data = df, x="month", c = cmap(90), y = "NOx", label = "NOx", linewidth  =2 )
    # Checking that the pollutant is in the columns before adding it to the plot
    if "PM2.5" in df_day.columns:
        sns.lineplot(data = df, x="month", c = cmap(10), y = "PM2.5", label = "PM2.5", linewidth  =2)
    if "O3" in df_day.columns:
        sns.lineplot(data = df, x="month", c = cmap(200), y = "O3", label = "O3", linewidth  =2)

    ax.set(title= "Concentration by Month")
    ax.legend(title = "Pollutant")
    ax.set(xlabel = "Day of the Week")
    ax.set(ylabel = "μg/m3")
    ax.locator_params(axis='y', nbins=4)

    # Taking spines off
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{station}_Monthly_Profiling.png")
    plt.close()

    # Fourth plot and Fifth plot: NOx Hourly profiling and Windrose

    f, ax = plt.subplots(figsize = (16,8))

    sns.lineplot(data = df[df["isweekday"] == "Weekday"], x="hour", c = cmap(90), y = "NOx", label = "Weekday", linewidth  =2 )
    sns.lineplot(data = df[df["isweekday"] == "Weekend"], x="hour", c = cmap(90), y = "NOx", linestyle = "dashed", label = "Weekend", linewidth  =2)
    ax.set(title= "NOx - Concentration by Hour and type of day")
    ax.set(ylabel = "NOx(μg/m3)")
    ax.set(xlabel = "Hour of the Day")
    ax.locator_params(axis='y', nbins=4)

    # Taking spines off
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{output_dir}/{station}_NOx_Hourly_Weekday_Weekend_Profiling.png")
    plt.close()


    #Windrose:

    if pollution_windrose == True:

        data = df[["wd", "ws", "NOx"]]

        # Activating R to plot the windrose
        pandas2ri.activate()


        # Importing R packages
        base = importr('base')
        openair = importr('openair')
        grdevices = importr('grDevices')  # For graphics device control
        RColorBrewer = importr('RColorBrewer')  # For color palettes
        lattice = importr('lattice')  # For theming

        r_mydata = pandas2ri.py2rpy(data)

        # Custom R script to plot the pollution rose
        r_script = """
            library(openair)
            library(lattice)
            library(RColorBrewer)

            # Custom function to plot the pollution rose with adjusted settings
            plot_pollution_rose <- function(data) {
                # Set color palette
                colors <- brewer.pal(6, "RdPu")

                # Plotting with custom theme adjustments
                pollutionRose(
                data,
                pollutant = "NOx",
                key.footer = "NOx",
                key.position = "right",
                key = TRUE,
                breaks = 6,
                paddle = FALSE,
                seg = 0.9,
                normalise = FALSE,
                alpha = 1,
                plot = TRUE,
                cols = colors,
                par.settings = list(
                    fontsize = list(text = 40, points = 16),  # Set the general text size
                    axis.line = list(col = "black"),
                    strip.background = list(col = "lightgrey"),
                    strip.border = list(col = "black")
                )
                )
            }
            # Call the custom plotting function
            plot_pollution_rose
            """

        # Loading the R script into Python
        robjects.r(r_script)

        # Reference the custom function from R environment
        plot_pollution_rose = robjects.globalenv['plot_pollution_rose']

        # Open a graphics device to save the plot
        grdevices.png(file=f"{output_dir}/{station}_NOx_Pollution_Rose.png", width=1600, height=1200, res=100)

        # Call the R function to create the plot
        plot_pollution_rose(r_mydata)

        # Close the graphics device to save the plot
        grdevices.dev_off()
        pandas2ri.deactivate()

    # Sixth plot and Seventh plot: PM2.5 Hourly profiling and Windrose (if available)

    if "PM2.5" in df.columns:
        f, ax = plt.subplots(figsize = (16,8))

        sns.lineplot(data = df[df["isweekday"] == "Weekday"], x="hour", c = cmap(10), y = "PM2.5", label = "Weekday", linewidth  =2 )
        sns.lineplot(data = df[df["isweekday"] == "Weekend"], x="hour", c = cmap(10), y = "PM2.5", linestyle = "dashed", label = "Weekend", linewidth  =2)
        ax.set(title= "PM2.5 - Concentration by Hour and type of day")
        ax.set(ylabel = "PM2.5(μg/m3)")
        ax.set(xlabel = "Hour of the Day")
        ax.locator_params(axis='y', nbins=4)

        # Taking spines off
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.savefig(f"{output_dir}/{station}_PM2_5_Hourly_Weekday_Weekend_Profiling.png")
        plt.close()


        #Windrose:

        if pollution_windrose == True:

            data = df[["wd", "ws", "PM2.5"]]

            # Activating R to plot the windrose
            pandas2ri.activate()


            # Importing R packages
            base = importr('base')
            openair = importr('openair')
            grdevices = importr('grDevices')  # For graphics device control
            RColorBrewer = importr('RColorBrewer')  # For color palettes
            lattice = importr('lattice')  # For theming

            r_mydata = pandas2ri.py2rpy(data)

            # Custom R script to plot the pollution rose
            r_script = r_script = """
                                library(openair)
                                library(lattice)
                                library(RColorBrewer)

                                # Custom function to plot the pollution rose with adjusted settings
                                plot_pollution_rose <- function(data) {
                                    # Set color palette
                                    colors <- brewer.pal(6, "Purples")

                                    # Plotting with custom theme adjustments
                                    pollutionRose(
                                    data,
                                    pollutant = "PM2.5",
                                    key.footer = "PM2.5",
                                    key.position = "right",
                                    key = TRUE,
                                    breaks = 6,
                                    paddle = FALSE,
                                    seg = 0.9,
                                    normalise = FALSE,
                                    alpha = 1,
                                    plot = TRUE,
                                    cols = colors,
                                    par.settings = list(
                                        fontsize = list(text = 40, points = 16),  # Set the general text size
                                        axis.line = list(col = "black"),
                                        strip.background = list(col = "lightgrey"),
                                        strip.border = list(col = "black")
                                    )
                                    )
                                }

                                # Call the custom plotting function
                                plot_pollution_rose
                                """

            # Loading the R script into Python
            robjects.r(r_script)

            # Reference the custom function from R environment
            plot_pollution_rose = robjects.globalenv['plot_pollution_rose']

            # Open a graphics device to save the plot
            grdevices.png(file=f"{output_dir}/{station}_PM2_5_Pollution_Rose.png", width=1600, height=1200, res=100)

            # Call the R function to create the plot
            plot_pollution_rose(r_mydata)

            # Close the graphics device to save the plot
            grdevices.dev_off()
            pandas2ri.deactivate()

    # Eigth plot and Ninth plot: O3 Hourly profiling and Windrose (if available)

    if "O3" in df.columns:
        f, ax = plt.subplots(figsize = (16,8))

        sns.lineplot(data = df[df["isweekday"] == "Weekday"], x="hour", c = cmap(200), y = "O3", label = "Weekday", linewidth  =2 )
        sns.lineplot(data = df[df["isweekday"] == "Weekend"], x="hour", c = cmap(200), y = "O3", linestyle = "dashed", label = "Weekend", linewidth  =2)
        ax.set(title= "O3 - Concentration by Hour and type of day")
        ax.set(ylabel = "O3(μg/m3)")
        ax.set(xlabel = "Hour of the Day")
        ax.locator_params(axis='y', nbins=4)

        # Taking spines off
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.savefig(f"{output_dir}/{station}_O3_Hourly_Weekday_Weekend_Profiling.png")
        plt.close()


        #Windrose:

        if pollution_windrose == True:

            data = df[["wd", "ws", "O3"]]

            # Activating R to plot the windrose
            pandas2ri.activate()


            # Importing R packages
            base = importr('base')
            openair = importr('openair')
            grdevices = importr('grDevices')  # For graphics device control
            RColorBrewer = importr('RColorBrewer')  # For color palettes
            lattice = importr('lattice')  # For theming

            r_mydata = pandas2ri.py2rpy(data)

            # Custom R script to plot the pollution rose
            r_script = r_script = """
                            library(openair)
                            library(lattice)
                            library(RColorBrewer)

                            # Custom function to plot the pollution rose with adjusted settings
                            plot_pollution_rose <- function(data) {
                                # Set color palette
                                colors <- brewer.pal(6, "Oranges")

                                # Plotting with custom theme adjustments
                                pollutionRose(
                                data,
                                pollutant = "O3",
                                key.footer = "O3",
                                key.position = "right",
                                key = TRUE,
                                breaks = 6,
                                paddle = FALSE,
                                seg = 0.9,
                                normalise = FALSE,
                                alpha = 1,
                                plot = TRUE,
                                cols = colors,
                                par.settings = list(
                                    fontsize = list(text = 40, points = 16),  # Set the general text size
                                    axis.line = list(col = "black"),
                                    strip.background = list(col = "lightgrey"),
                                    strip.border = list(col = "black")
                                )
                                )
                            }

                            # Call the custom plotting function
                            plot_pollution_rose
                            """

            # Loading the R script into Python
            robjects.r(r_script)

            # Reference the custom function from R environment
            plot_pollution_rose = robjects.globalenv['plot_pollution_rose']

            # Open a graphics device to save the plot
            grdevices.png(file=f"{output_dir}/{station}_O3_Pollution_Rose.png", width=1600, height=1200, res=100)

            # Call the R function to create the plot
            plot_pollution_rose(r_mydata)

            # Close the graphics device to save the plot
            grdevices.dev_off()
            pandas2ri.deactivate()

    print(f"✅ Station {station} EDA completed.")
    print("")

output_dir = f"EDA_results/datasets"
os.makedirs(output_dir, exist_ok=True)

# Saving the combined DataFrame for weather normalisation process
combined_df[pollutants + ["wd","ws","temp", "station"]].to_csv("EDA_results/datasets/combined_AURN_data.csv")

print("✅✅✅✅✅✅ EDA RESULTS COMPLETED ✅✅✅✅✅✅")
