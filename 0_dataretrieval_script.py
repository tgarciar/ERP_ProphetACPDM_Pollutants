### THIS SCRIPT RUNS THE TWO SCRIPTS USED FOR AURN DATA RETRIEVAL (FOUND IN data_retrival FOLDER)

################################################################################


# Importing the default function to run the scripts
from others.extra_pyfunctions.functions import run_script

# Defining the working directory containing both scripts (download_AURNgovdata.py and merging_stations_AURNgovdata.py)
working_dir = "data_retrieval"

# Defining the names of the of scripts that will be run
download_AURN_script = "01_download_AURNgovdata.py"
merging_AURN_script = "02_merging_stations_AURNgovdata.py"

# Running the download_AURNgovdata.py script
download_success = run_script(download_AURN_script, working_dir)

# Running the merging_stations_AURNgovdata.py script only if the first script was successful
if download_success:
    merging_success = run_script(merging_AURN_script, working_dir)
else:
    merging_success = False

# Printing final message all scripts were successful!
if download_success and merging_success:
    print("✅✅✅✅✅✅ DATA READY FOR EDA ✅✅✅✅✅✅")
