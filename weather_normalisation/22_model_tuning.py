### IMPORTANT: THIS SCRIPT IS COMPUTATIONTALLY INTENSIVE. IT TOOK AROUND 40 HOURS TO RUN IT ON GOOGLE COLAB WITH AN L4 GPU.
### BY DEFAULT, THIS SCRIPT WILL BE SKIPPED. IF THE USER WANT TO RUN IT, NEED TO CHANGE model_tuning = TRUE ON 1_weather_normalisation.py

### THIS SCRIPT WILL RUN ALL THE MODEL TUNING FROM THE WEATHER NORMALISATION PROCESS EXPLAINED IN THE REPORT. IT WILL SAVE THE RESULT IN weather_normalisation/personalised_results
### IT WILL:
### 1. Run a NORMET weather normalisation, save the best model/parameters. The results will be used in 23_resampling_ts.py
### 2. Model Tuning of a XGBooster model with RandomizedSearchCV (5 Time Series Cross validation / w. all the data)--> 200 iterations. Save all the results. The results will be used in 23_resampling_ts.py
### 3. Model Tuning of a RandomForest model with RandomizedSearchCV (5 Time Series Cross validation / w. all the data)--> 200 iterations. Save all the results. The results will be used in 23_resampling_ts.py


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  RandomizedSearchCV
import normet
import warnings

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


############### PARAMETERS ####################################################

# Parameters to run the script. This can be changed by the user.

# PARAMETERS TO RUN:

pollutants_list = ["NOx",
                   "PM2.5",
                   "O3"
                   ]
station_list = [
    "Manchester_Piccadilly",
    'Manchester_Sharston',
    "Salford_Eccles",
    "Salford_Glazebury",
    "Bury_Whitefield"
    ]


# Features to use in Normet, XGBooster and Random Forest models training/tuning:
feature_names=['temp', 'RH', 'sp', 'blh_final', 'weekday',
               'month_sin', 'month_cos', 'wind_u', 'wind_v', 'hour_sin',
               'hour_cos', 'day_sin', 'day_cos', 'unix']


## STEP 1: Normet parameters
model_config = {
'time_budget': 90,                                    # Total running time in seconds
'metric': "r2",                                       # Primary metric for regression, 'mae', 'mse', 'r2', 'mape',...
'estimator_list': ["rf","xgboost"],                   # List of ML learners: ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"]
'task': 'regression'                                  # Task type
}

split_method = "random"                               # How to separate the data into training and testing: Should be "random" or "ts"

## STEP 2 AND 3: RandomizedSearchCV parameters

number_iterations = 200                          # Number of iterations to search in RandomizedSearchCV
scoring ='r2'                                   # Scoring metric to search in RandomizedSearchCV

# XGB_Booster -  RandomizedSearchCV: STEP 2 PARAMETERS
device_xgb = "cpu"                                    # "cpu" or "gpu" . If "gpu" is selected, the user needs to have a GPU compatible with XGBoost

random_search_grid_xgb = {                            # Params to search
      'n_estimators': np.arange(100, 501, 5),
      'max_depth': np.arange(5, 75, 5),
      'max_leaves': np.arange(100, 1001, 10),
      'min_child_weight': np.logspace(-3, 1, 500),
      'learning_rate': np.logspace(-3, 0, 500),
      'subsample': np.linspace(0.5, 1.0, 20),
      'colsample_bylevel': np.linspace(0.5, 1.0, 20),
      'colsample_bytree': np.linspace(0.5, 1.0, 20),
      'reg_alpha': np.logspace(-3, 4, 100),
      'reg_lambda': np.logspace(-3, 4, 100)
  }

# RF Model -  RandomizedSearchCV: STEP 3 PARAMETERS


random_search_grid_rf = {                            # Params to search
    'n_estimators': np.arange(50, 501, 5),
    'max_features': ['sqrt', 'log2'],
    'max_depth': np.arange(5, 75, 5),
    'min_samples_split': np.arange(2, 21, 1),
    'min_samples_leaf': np.arange(1, 21, 1),
    'bootstrap': [True],
}

# Creating directory to save all the results
output_dir = f"personalised_weather_normalisation_results/model_tuning/"
os.makedirs(output_dir, exist_ok=True)


###############################################

#  Preparing Function to run the Model Tuning:

# NORMET_AUTOML MODEL TUNING - Function
def normet_automl_(data, station , pollutant , feature_names = feature_names, model_config = model_config, split_method = split_method):
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  print("STEP 1: NORMET weather normalisation, saving the best model/parameters.")
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


  # Preparing the dataset
  df = normet.prepare_data(data, value = pollutant, feature_names = feature_names, split_method=split_method,  fraction=0.8, seed=42)

  # Running AutoML
  normet_automl=normet.train_model(df ,variables = feature_names, model_config= model_config )

  # Best model
  best_model = normet_automl.best_estimator
  best_config = normet_automl.best_config
  testing_R2 =  normet.modStats(df,normet_automl, set = "testing")["R2"][0]

  # Saving best model information in dataframe:
  best_model_df = pd.DataFrame({ "Station": [station], "Pollutant": [pollutant], "Method": ["NORMET AutoML"], "Best Model" : [best_model] , "Best Config" : [best_config] , "Testing R2":  [testing_R2] })

  return best_model_df

def weather_norm_xgb(data, station , pollutant, feature_names = feature_names, random_search_grid_xgb = random_search_grid_xgb, number_iterations = number_iterations, scoring = scoring, device = device_xgb):

  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  print("STEP 2: Model Tuning of XGBooster models with RandomizedSearchCV (5 Time Series Cross validation / w. all the data)")
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

  print("")
  print("")

  start_time = time.time()

  X = data[feature_names]
  y = data[pollutant]


  # Initialising the XGBoost model
  if device == "gpu":
    xgb = XGBRegressor(random_state=42, tree_method='gpu_hist')
  elif device == "cpu":
    xgb = XGBRegressor(random_state=42)

  # Initialising RandomizedSearchCV
  random_search = RandomizedSearchCV(estimator=xgb, param_distributions=random_search_grid_xgb,
                                    n_iter=number_iterations, cv=5, scoring=scoring, n_jobs=-1, random_state=42, verbose = 1)

  # Fitting RandomizedSearchCV
  random_search.fit(X, y)

  # Getting the best parameters
  best_params = random_search.best_params_


  # Saving the results in a DataFrame
  ran_search_results = pd.DataFrame(random_search.cv_results_) [['params',"mean_test_score","rank_test_score"]]
  ran_search_results["Station"] = station
  ran_search_results["Pollutant"] = pollutant
  ran_search_results["Model"] = "xgboost"

  print("")
  print(f"Best parameters found: {best_params}")
  print("")
  end_time = time.time()
  execution_time = end_time - start_time
  print(f"Execution time: {execution_time:.2f} seconds")
  print("")
  print("")

  return ran_search_results


def weather_norm_rf(data, station , pollutant, feature_names = feature_names, random_search_grid_rf = random_search_grid_rf,  number_iterations = number_iterations, scoring = scoring):

  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  print("STEP 3: Model Tuning of RandomForest models with RandomizedSearchCV (5 Time Series Cross validation / w. all the data)")
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

  print("")
  print("")

  start_time = time.time()

  X = data[feature_names]
  y = data[pollutant]


  # Initialising the RandomForest model
  rf = RandomForestRegressor(random_state=42)


  # Initialising RandomizedSearchCV
  random_search = RandomizedSearchCV(estimator= rf, param_distributions=random_search_grid_rf,
                                    n_iter=number_iterations, cv=5, scoring=scoring, n_jobs=-1, random_state=42, verbose = 1)

  # Fitting RandomizedSearchCV
  random_search.fit(X, y)

  # Getting the best parameters
  best_params = random_search.best_params_


# Saving the results in a DataFrame
  ran_search_results = pd.DataFrame(random_search.cv_results_) [['params',"mean_test_score","rank_test_score"]]
  ran_search_results["Station"] = station
  ran_search_results["Pollutant"] = pollutant
  ran_search_results["Model"] = "randomforest"

  print("")
  print(f"Best parameters found: {best_params}")
  print("")
  end_time = time.time()
  execution_time = end_time - start_time
  print(f"Execution time: {execution_time:.2f} seconds")
  print("")
  print("")

  return ran_search_results


############################## RUNING THE FOR LOOP FOR EVERY STATION AND POLLUTANT ##################

# Loading data:
data = pd.read_csv("datasets/data_for_wetnor.csv")
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S", errors = "raise")
data.set_index("date", inplace=True)


# Creating Dataframes to save results:
normet_bestmodel = pd.DataFrame(columns=["Station", "Pollutant",  "Method", "Best Model" , "Best Config" , "Testing R2" ])
ran_search_results = pd.DataFrame(columns=['params',"mean_test_score","rank_test_score","Station", "Pollutant",  "Model"])

### RUNING THE FOR LOOP FOR EVERY POLLUTANT AND STATION
for station in station_list:
  print(f"STARTING PROCESS WITH STATION: {station}")
  print("")
  print("")

  for pollutant in pollutants_list:

    print(f"STARTING PROCESS WITH POLLUTANT: {pollutant}")
    print("")

    # Filtering the data
    temp_data = data[data["station"] == station].copy()

    temp_data = temp_data[[pollutant] + feature_names ]

    #  Validating the pollutant data exists in the station (90% data not null as a threshold). If not, jump to next one.
    if len(temp_data.dropna()) > (0.9 * len(temp_data)):

      #Step 1:
      best_model_df  = normet_automl_(data = temp_data, pollutant = pollutant, station = station)

      normet_bestmodel = pd.concat([normet_bestmodel, best_model_df], axis=0, ignore_index=True)

      print("")
      print("")
      print("---------------------------------------------------------------------")
      print("✅ STEP 1: NORMET AUTOML PROCESS SUCCESSFULL...")
      print("---------------------------------------------------------------------")
      print("")
      print("")



      #Step 2:

      ran_search_res_xgb = weather_norm_xgb(data = temp_data, pollutant = pollutant, station = station)

      ran_search_results = pd.concat([ran_search_results, ran_search_res_xgb], axis=0, ignore_index=True)

      print("")
      print("")
      print("---------------------------------------------------------------------")
      print("✅ STEP 2: RANDOMIZEDSEARCHCV WITH XGBOOSTER PROCESS SUCCESSFULL...")
      print("---------------------------------------------------------------------")
      print("")
      print("")

      #Step 3:

      ran_search_res_rf = weather_norm_rf(data = temp_data, pollutant = pollutant, station = station)

      ran_search_results = pd.concat([ran_search_results, ran_search_res_rf], axis=0, ignore_index=True)


      print("")
      print("")
      print("---------------------------------------------------------------------")
      print("✅ STEP 3: RANDOMIZEDSEARCHCV WITH RANDOMFOREST PROCESS SUCCESSFULL...")
      print("---------------------------------------------------------------------")
      print("")
      print("")


      print("")
      print("---------------------------------------------------------------------")
      print("")
      print(f"                 ✅ {pollutant} PROCESS FINISHED!")
      print("")
      print("---------------------------------------------------------------------")
      print("")
      print("")


    else:
      print(f'{pollutant} NOT FOUND IN THIS STATION. STARTING WITH THE NEXT ONE...')
  print("")
  print("")
  print("---------------------------------------------------------------------")
  print("")
  print(f"            ✅ {station} PROCESS FINISHED!")
  print("")
  print("---------------------------------------------------------------------")
  print("")
  print("")

#Saving Results:

# All Models - Cross_validations:

ran_search_results.sort_values(["rank_test_score","Station","Pollutant", "mean_test_score"], ascending=  [True,True,True,False] ,inplace = True)
ran_search_results.reset_index(drop = True, inplace=True)
ran_search_results.to_csv(f'{output_dir}/results_randomsearch_allmodels.csv', index = False)

#Auto-ML Best models:
normet_bestmodel.to_csv(f'{output_dir}/normet_bestmodels.csv', index = False)

print(f" ✅✅✅ MODEL TUNING PROCESS FINISHED! RESULTS CAN BE FIND IN {output_dir}")
