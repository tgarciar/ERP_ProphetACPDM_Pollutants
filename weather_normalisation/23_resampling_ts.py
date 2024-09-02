### IMPORTANT: THIS SCRIPT IS COMPUTATIONAL INTENSIVE. WHEN DOING THE RESEARCH IT TOOK 10 HOURS TO RUN. BY DEFAULT,
### THIS SCRIPT WILL BE SKIPPED WHEN RUNING 2_weather_normalisation.py, IF THE USER WANT TO RUN IT, PLEASE CHANGE TO resampling = True ON 2_weather_normalisation.py SCRIPT
### FOR CHECKING HOW IT WORKS, IT IS RECOMMENDED TO RUN IT ON ONE STATION, POLLUTANT (MANCHESTER_PICADILLY, NOx) AND RESAMPLE ONLY 5/10 TIMES. (PARAMETERS ON LINES 53-80)

### THIS SCRIPT WILL RESAMPLE THE DATA USING ALLVAR AND METVAR APPROACHES AND WEATHER NORMALISED THE TIME SERIES.
### FOR THIS, THE BEST MODEL WILL BE TRAINED BASED ON THE MODEL TUNING RESULTS. BY DEFAULT, THE RESULTS FROM THE ERP WILL BE USED.
### THIS CAN BE CHANGED BY SETTING ERP_results = False IN LINE 14. THE MODEL TRAINED WILL BE SAVED IN trained_models FOLDER.
### THE WETNOR TIME SERIES WILL BE SAVED IN THE datasets/WETNOR FOLDER.


################################################################################
#PARAMETERS THAT CAN BE CHANGED:

# Checking if MODEL TUNING is defined. If not, it will be set to False
try:
   model_tuning = os.environ['MODEL_TUNING']
except KeyError:
   model_tuning = "False"                             # USER CAN MANUALLY CHANGE THIS TO USE PERSONALISED RESULTS. FALSE MEANS THAT IT WILL USE THE ERP_RESULTS, TRUE MEANS IT WILL USE THE RESULTS OF THE PERSONALISED MODEL_TUNING (22_model_tuning.py)


#################################################################################


# Importing libraries:
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import pickle
import ast
import warnings

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


# Loading the dataset ready for the weather normalisation

data_for_wetnor = pd.read_csv("datasets/data_for_wetnor.csv")
data_for_wetnor["date"] = pd.to_datetime(data_for_wetnor["date"], format="%Y-%m-%d %H:%M:%S", errors = "raise")
data_for_wetnor.set_index("date", inplace=True)



# Loading the results
if model_tuning == "False":
    RS_results_bestmodels = pd.read_csv("saved_results/model_tuning/results_randomsearch_allmodels.csv")
    NORMET_results_bestmodels = pd.read_csv("saved_results/model_tuning/normet_bestmodels.csv")
elif model_tuning == "True":
    RS_results_bestmodels = pd.read_csv("personalised_weather_normalisation_results/model_tuning/results_randomsearch_allmodels.csv")
    NORMET_results_bestmodels = pd.read_csv("personalised_weather_normalisation_results/model_tuning/normet_bestmodels.csv")


RS_results_bestmodels.reset_index(drop=True, inplace=True)
NORMET_results_bestmodels.reset_index(drop=True, inplace=True)




########### PARAMETERS USED FOR TRAINING MODELS, RESAMPLING AND WEATHER NORMALISATION #########

stations_list = ["Manchester_Piccadilly",
                 "Manchester_Sharston",
                 "Salford_Eccles",
                 "Salford_Glazebury",
                 "Bury_Whitefield"
                 ]            # List of stations

pollutants =  ["NOx",
               "PM2.5",
               "O3"
               ]             # List of pollutants

features_names= ['temp', 'RH', 'sp', 'blh_final', 'weekday',                   # Features for training of the models:
               'month_sin', 'month_cos', 'wind_u', 'wind_v', 'hour_sin',
               'hour_cos', 'day_sin', 'day_cos', 'unix']


# Resampling Methods features:
ALLVAR_resampling_features = ['temp', 'RH', 'sp', 'blh_final', 'weekday',       # Features for ALLVAR resampling:
               'month_sin', 'month_cos', 'wind_u', 'wind_v', 'hour_sin',
               'hour_cos', 'day_sin', 'day_cos']

METVAR_resampling_features = ['temp', 'RH', 'sp', 'blh_final', 'wind_u', 'wind_v'] # Features for METVAR resampling

n_resampling = 1000                                                           # Number of resampling









##################################################################################
# Normalising Function. This function will normalise the time series using the best model for each station and pollutant
def normalise_ts(X, model, resample_columns, num_resampling=n_resampling):
  print("")
  print("Resampling for weather normalisation...")

  predictions = np.zeros((len(X), num_resampling))

  for i in range(num_resampling):
       sampled_data = X.copy()
       # Shuffleing values within each specified column without replacement (To not introduce bias)
       for column in resample_columns:
           sampled_data[column] = sampled_data[column].sample(n=len(X), replace=False, random_state=i).values
       # Predicting Pollutant concentrations using the model with the partially shuffled data
       predictions[:, i] = model.predict(sampled_data)

   # Calculating the mean of predictions
  normalised_ts = predictions.mean(axis=1)

  return normalised_ts







######################### RUNING THE FOR LOOP FOR EACH STATION AND POLLUTANT ############################

# Creating directories to save the models:
output_dir_models_rs = f"models/random_search/"
os.makedirs(output_dir_models_rs, exist_ok=True)

output_dir_models_normet = f"models/normet/"
os.makedirs(output_dir_models_normet, exist_ok=True)


# Creating Dataframes to store the results
results_best_models = pd.DataFrame(columns = ["station", "pollutant", "type","model","test_r2", "train_r2", "full_r2","test_mse", "train_mse", "full_mse", "params" ])
feature_importance_best_models = pd.DataFrame(columns = ["station","pollutant","model","feature","importance"])
combined_WETNOR_ds = pd.DataFrame(columns = ["measure","station","pollutant","model_used","ALLVAR","METVAR","ALLVAR_normet","METVAR_normet"])

for station in stations_list:
    for pollutant in pollutants:
        print("")
        print(f"Normalising {pollutant} for {station}")
        print("")
        print("")
        model = None
        model_normet = None

        temp_data = data_for_wetnor[data_for_wetnor["station"] == station].copy()

        temp_data = temp_data[[pollutant] + features_names]

        if len(temp_data.dropna()) > (0.9 * len(temp_data)):

            # Selecting the best model (RANDOM SEARCH) for the station and pollutant
            X = temp_data[features_names]
            y = temp_data[pollutant]

            X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state=42)


            ########## RANDOM SEARCH BEST MODEL ##########
            print("")
            print("RandomSearch BestModel Training and Resampling...")
            print("")
            # Selecting Best Result and Best Model:
            filtered_results = RS_results_bestmodels[(RS_results_bestmodels["Pollutant"] == pollutant) & (RS_results_bestmodels["Station"] == station) & (RS_results_bestmodels["rank_test_score"] == 1)]

            selected_model = filtered_results.loc[filtered_results["mean_test_score"].idxmax()]["Model"]

            # Selecting params of best model. The
            params = filtered_results[filtered_results["Model"] == selected_model].iloc[0]["params"]
            params = ast.literal_eval(params)

            if selected_model == "xgboost":
                model = XGBRegressor(**params, random_state=42)
            elif selected_model == "randomforest":
                model = RandomForestRegressor(**params, random_state=42)

            #Training the model
            model.fit(X_train,y_train)

            # Predicting for results:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred = model.predict(X)

            # Calculating R^2
            train_r2 = r2_score(y_train, y_pred_train)

            test_r2 = r2_score(y_test, y_pred_test)

            full_r2 = r2_score(y, y_pred)


            # Calculating Mean Squared Error
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            full_mse = mean_squared_error(y, y_pred)

            print(f"Training:")
            print(f"R^2: {train_r2}")
            print(f"MSE: {train_mse}")
            print("")
            print(f"Test:")
            print(f"R^2: {test_r2}")
            print(f"MSE: {test_mse}")
            print()
            print(f"Full:")
            print(f"R^2: {full_r2}")
            print(f"MSE: {full_mse}")
            print("")

            # Saving Results:
            results_temp = {"station": station,
                "pollutant": pollutant,
                "model": selected_model,
                "type": "Random Search",
                "test_r2": test_r2,
                "train_r2": train_r2,
                "full_r2": full_r2,
                "test_mse": test_mse,
                "train_mse": train_mse,
                "full_mse": full_mse,
                "params": f'{params}'
                }

            #Saving results of best models
            results_best_models = pd.concat([results_best_models,pd.DataFrame([results_temp])])

            #Saving Feature Importance
            features_in = list(model.feature_names_in_)
            features_importance = list(model.feature_importances_)

            df_feature_importance = pd.DataFrame({
                                                'feature': features_in,
                                                 'importance': features_importance
                                                })

            df_feature_importance["model"] = selected_model
            df_feature_importance["station"] = station
            df_feature_importance["pollutant"] = pollutant
            df_feature_importance["type"] = "Random Search"

            feature_importance_best_models = pd.concat([feature_importance_best_models, df_feature_importance])

            # Saving the model

            with open(f'{output_dir_models_rs}/{station}_{pollutant}_randomsearch_bestmodel.pkl', 'wb') as file:
                pickle.dump(model, file)

            # Resampling and Weather Normalising the Time Series:

            # Preparing Dataframe for saving results
            WETNOR_ds = temp_data[[pollutant]].copy()
            WETNOR_ds.rename(columns = {pollutant: "measure"}, inplace = True)
            WETNOR_ds["station"] = station
            WETNOR_ds["pollutant"] = pollutant
            WETNOR_ds["model_used"] = selected_model

            #ALLVAR RESAMPLING
            WETNOR_ds["ALLVAR"] = normalise_ts(X, model, ALLVAR_resampling_features)

            #METVAR RESAMPLING
            WETNOR_ds["METVAR"] = normalise_ts(X, model, METVAR_resampling_features)



            ########## NORMET_AUTOML BEST MODEL ##########
            print("")
            print("NORMET_AUTOML BestModel Training and Resampling...")
            print("")

            # Selecting Best Result and Best Model:
            filtered_results = NORMET_results_bestmodels[(NORMET_results_bestmodels["Pollutant"] == pollutant) & (NORMET_results_bestmodels["Station"] == station)]


            selected_model = filtered_results.iloc[0]["Best Model"]

            # Selecting params of best model. The
            params = filtered_results.iloc[0]["Best Config"]
            params = ast.literal_eval(params)

            if selected_model == "xgboost":
                model_normet = XGBRegressor(**params, random_state = 42)
            elif selected_model == "randomforest":
                model_normet = RandomForestRegressor(**params, random_state=42)

            #Training the model
            model_normet.fit(X_train,y_train)

            # Predicting for results:
            y_pred_train = model_normet.predict(X_train)
            y_pred_test = model_normet.predict(X_test)
            y_pred = model_normet.predict(X)

            # Calculating R^2
            train_r2 = r2_score(y_train, y_pred_train)

            test_r2 = r2_score(y_test, y_pred_test)

            full_r2 = r2_score(y, y_pred)


            # Calculating Mean Squared Error
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            full_mse = mean_squared_error(y, y_pred)

            print(f"Training:")
            print(f"R^2: {train_r2}")
            print(f"MSE: {train_mse}")
            print("")
            print(f"Test:")
            print(f"R^2: {test_r2}")
            print(f"MSE: {test_mse}")
            print()
            print(f"Full:")
            print(f"R^2: {full_r2}")
            print(f"MSE: {full_mse}")
            print("")

            # Saving Results:
            results_temp = {"station": station,
                "pollutant": pollutant,
                "model": selected_model,
                "type": "Normet AutoML",
                "test_r2": test_r2,
                "train_r2": train_r2,
                "full_r2": full_r2,
                "test_mse": test_mse,
                "train_mse": train_mse,
                "full_mse": full_mse,
                "params": f'{params}'
                }

            #Saving results of best models
            results_best_models = pd.concat([results_best_models,pd.DataFrame([results_temp])])

            #Saving Feature Importance
            features_in = list(model_normet.feature_names_in_)
            features_importance = list(model_normet.feature_importances_)

            df_feature_importance = pd.DataFrame({
                                                'feature': features_in,
                                                 'importance': features_importance
                                                })

            df_feature_importance["model"] = selected_model
            df_feature_importance["station"] = station
            df_feature_importance["pollutant"] = pollutant
            df_feature_importance["type"] = "Normet AutoML"

            feature_importance_best_models = pd.concat([feature_importance_best_models, df_feature_importance])


            # Saving the model

            with open(f'{output_dir_models_normet}/{station}_{pollutant}_normet_bestmodel.pkl', 'wb') as file:
                pickle.dump(model_normet, file)

            # Resampling and Weather Normalising the Time Series:

            #ALLVAR RESAMPLING
            WETNOR_ds["ALLVAR_normet"] = normalise_ts(X, model_normet, ALLVAR_resampling_features)

            #METVAR RESAMPLING
            WETNOR_ds["METVAR_normet"] = normalise_ts(X, model_normet, METVAR_resampling_features)


            combined_WETNOR_ds = pd.concat([combined_WETNOR_ds, WETNOR_ds])



# Saving the results:

output_dir_wetnor = f"personalised_weather_normalisation_results/WETNOR_timeseries"
os.makedirs(output_dir_wetnor, exist_ok=True)

combined_WETNOR_ds.index.name = 'date'
combined_WETNOR_ds.to_csv(f"{output_dir_wetnor}/personalised_WETNOR_combined_ts.csv", index=True)


output_dir_pers_model_training = f"personalised_weather_normalisation_results/model_training"
os.makedirs(output_dir_pers_model_training, exist_ok=True)

feature_importance_best_models.to_csv(f"{output_dir_pers_model_training}/feature_importance_best_models.csv", index=False)
results_best_models.to_csv(f"{output_dir_pers_model_training}/results_best_models.csv", index=False)


print("✅✅✅ ALL RESAMPLES FINISHED. RESULTS SAVED IN personalised_weather_normalisation ")
