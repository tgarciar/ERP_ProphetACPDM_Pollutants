### THIS SCRIPT RUNS ALL THE SCRIPTS FOUND ON experiment1/

# Importing the default function to run the scripts
from others.extra_pyfunctions.functions import run_script

# Importing OS to save the variable
import os
import sys
################################################################################
# PARAMS THAT CAN BE CHANGED:

# IMPORTANT: 31_experiment1.py PROCESS MAY TAKE MORE THAN 2 HOURS TO RUN.
# FOR THIS REASON, BY DEFAULT, WE LOAD THE RESULTS FROM THE RESEARCH.
# IF "load_results" = "True", IT WILL RUN 32_experiment1_plots.py ONLY.
# IF "load_results" = "False", IT WILL RUN 31_experiment1.py BEFORE PLOTTING THE RESULTS.

os.environ['load_results'] = "True"  # "True" (DEFAULT) or "False"

os.environ['ERP_DATA'] = "True"  # "True" (DEFAULT) or "False" # USING DATA SAVED FROM THE WETNOR_ERP IN 31_experiment1.py
################################################################################

# Defining the working directory containing both scripts
working_dir = "experiment1"

# Defining the names of the scripts that will be run
experiment_script = "31_experiment1.py"
plotting_script = "32_experiment1_plots.py"

# Conditional logic based on the environment variable:
if os.environ['load_results'] == "True":
    # If load_results is True, only run the plotting script
    plot_script = run_script(plotting_script, working_dir)
    if plot_script:
        print("✅✅✅✅✅✅ EXPERIMENT 1 PLOTTING DONE ✅✅✅✅✅✅")
    else:
        print("❌❌❌❌❌❌ PLOTTING FAILED ❌❌❌❌❌❌")
elif os.environ['load_results'] == "False":
    # If load_results is False, run both the experiment script and then the plotting script
    experiment_run = run_script(experiment_script, working_dir)
    if experiment_run:
        plot_script = run_script(plotting_script, working_dir)
        if plot_script:
            print("✅✅✅✅✅✅ EXPERIMENT 1 COMPLETE ✅✅✅✅✅✅")
        else:
            print("❌❌❌❌❌❌ PLOTTING FAILED ❌❌❌❌❌❌")
    else:
        print("❌❌❌❌❌❌ EXPERIMENT 1 FAILED ❌❌❌❌❌❌")
else:
    print("❌ Process Shutdown. Invalid setting for load_results. Please use 'True' or 'False'.")
    sys.exit(1)  # Non-zero value indicates an error
