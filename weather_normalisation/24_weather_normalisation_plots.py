### THIS SCRIPT WILL PRODUCE THE PLOTS PRESENTED IN THE PAPER AND THE TABLE (AS CSV) PRESENTED IN THE PAPER


import os

####################################################################
#PARAMETERS THAT CAN BE CHANGED:
try:
    resampling = os.environ['RESAMPLING']
except KeyError:
   resampling = "False"           # USER CAN MANUALLY CHANGE THIS TO USE PERSONALISED RESULTS. IF FALSE, IT WILL USE THE ERP_RESULTS. IF TRUE, IT WILL USE THE RESULTS OF THE PERSONALISED (23_resampling_ts.py)


####################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Opening the csv if
if resampling == "False":
    results_bestmodels = pd.read_csv("saved_results/model_training/results_best_models.csv")
    feature_importance = pd.read_csv("saved_results/model_training/feature_importance_best_models.csv")
elif resampling == "True":
    results_bestmodels = pd.read_csv("personalised_weather_normalisation_results/model_training/results_best_models.csv")
    feature_importance = pd.read_csv("personalised_weather_normalisation_results/model_training/feature_importance_best_models.csv")


#Changing names of the Features for more interpretability:
feature_importance["feature"] = feature_importance["feature"].replace("temp","Temperature", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("RH","RelativeHumidity", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("sp","SurfacePressure", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("blh_final","BLH", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("weekday","Weekday", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("month_sin","Month(Sin)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("month_cos","Month(Cos)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("wind_u","Wind(u)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("wind_v","Wind(v)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("hour_sin","Hour(Sin)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("hour_cos","Hour(Cos)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("day_sin","Day(Sin)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("day_cos","Day(Cos)", regex = True)
feature_importance["feature"] = feature_importance["feature"].replace("unix","UNIX", regex = True)


#Creating directory for saving results
output_dir = f"weather_normalisation_plots/"
os.makedirs(output_dir, exist_ok=True)


# Setting customs for plotting:
custom_palette = ["darkviolet", "orchid"]
sns.set_context("paper",font_scale=1.7)

f, ax = plt.subplots(figsize = (16,12))

sns.despine(bottom=True, left=True)

sns.stripplot(
    data=feature_importance, x="importance", y="feature", hue="type",
    dodge=True, alpha=.20, zorder=1, legend=False, palette = custom_palette, s = 7,
)
sns.pointplot(
    data=feature_importance, x="importance", y="feature", hue="type",
    dodge=.8 - .8 / 3, errorbar = "ci",
    markers="d", markersize=7, linestyle="none",  palette=custom_palette
)

ax.set(ylabel = "Features")
ax.set(xlabel = "Importance")
ax.legend(title = "Method")


plt.savefig(f"{output_dir}/Feature_Importance_models_trained.png")
plt.close()

sns.set_context("paper",font_scale=1.7)
sns.set_style("white")

f, ax = plt.subplots(figsize = (16,12))

ax = sns.swarmplot(data=results_bestmodels, x="pollutant", y="test_r2", hue="type", palette = custom_palette, marker = "d", legend = True, s = 10)
ax.set(ylabel="Test R2")
ax.set(xlabel="Pollutant")
ax.set(ylim = (0.40,0.9))

ax.legend(title = "Method")

for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig(f"{output_dir}/R2_score_models_trained.png")
plt.close()

# Creating CSV with the results:
pivot_df = round(results_bestmodels.pivot(index=['station', 'pollutant'], columns='type', values='test_r2'),2).reset_index()
pivot_df.to_csv(f"{output_dir}/R2_scores.csv")

print("✅✅✅ RESULTS SAVED IN weather_normalisation_paper_results")
