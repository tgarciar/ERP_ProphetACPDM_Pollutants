### THIS SCRIPT RUNS ALL THE SCRIPTS FOUND ON weather_normalisation

# Importing the default function to run the scripts
from others.extra_pyfunctions.functions import run_script

# Importing OS to save the variable
import os
import sys
################################################################################
# PARAMS THAT CAN BE CHANGED:

# This two process may take more than 50 hours to run. For this reasons, by DEFAULT
# we will skip them and use the results from the ERP (weather_normalisation/ERP_weather_normalisation_results)
# For changing specific parameters of the 22_model_tuning.py and 23_resampling_ts.py, please refer to the scripts.

# This can be changed by the user. Default = "False".

os.environ['MODEL_TUNING'] = "False"                    #"False"(DEFAULT) or "True"
os.environ['RESAMPLING'] = "False"                      #"False"(DEFAULT) or "True"


################################################################################

# Defining the working directory containing both scripts
working_dir = "weather_normalisation"

# Defining the names of the of scripts that will be run
feature_engineering_script = "21_feature_engineering.py"
model_tuning_script = "22_model_tuning.py"
resampling_script = "23_resampling_ts.py"
paper_results_script = "24_weather_normalisation_plots.py"

# Running the scripts:
first_script = run_script(feature_engineering_script, working_dir)

# Conditional logic based on the environment variables:
if os.environ['RESAMPLING'] == "False" and os.environ['MODEL_TUNING'] == "False":
    if first_script:
        second_script = run_script(paper_results_script, working_dir)
    else:
        second_script = False

elif os.environ['MODEL_TUNING'] == "True" and os.environ['RESAMPLING'] == "False":
    if first_script:
        second_script = run_script(model_tuning_script, working_dir)
        if second_script:
            third_script = run_script(paper_results_script, working_dir)
        else:
            third_script = False
    else:
        second_script = False
        third_script = False

elif os.environ['MODEL_TUNING'] == "True" and os.environ['RESAMPLING'] == "True":
    if first_script:
        second_script = run_script(model_tuning_script, working_dir)
        if second_script:
            third_script = run_script(resampling_script, working_dir)
            if third_script:
                fourth_script = run_script(paper_results_script, working_dir)
            else:
                fourth_script = False
        else:
            third_script = False
            fourth_script = False
    else:
        second_script = False
        third_script = False
        fourth_script = False
else:
    print(" ❌ Process Shutdown. Something is wrong with the setting OS.ENVIRON variables... CAN'T BE TUNING = 'False' AND RESAMPLING = 'True'")
    sys.exit(1)  # Non-zero value indicates an error

# Final message if all scripts were successful
if os.environ['MODEL_TUNING'] == "False" and os.environ['RESAMPLING'] == "False":
    if first_script and second_script:
        print("✅✅✅✅✅✅ WEATHER NORMALISATION DONE ✅✅✅✅✅✅")
elif os.environ['MODEL_TUNING'] == "True" and os.environ['RESAMPLING'] == "False":
    if first_script and second_script and third_script:
        print("✅✅✅✅✅✅ WEATHER NORMALISATION DONE ✅✅✅✅✅✅")
elif os.environ['MODEL_TUNING'] == "True" and os.environ['RESAMPLING'] == "True":
    if first_script and second_script and third_script and fourth_script:
        print("✅✅✅✅✅✅ WEATHER NORMALISATION DONE ✅✅✅✅✅✅")
