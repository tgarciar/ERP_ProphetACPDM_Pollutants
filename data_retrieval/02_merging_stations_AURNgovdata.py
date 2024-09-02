### THIS SCRIPT WILL MERGE THE YEARLY AURN RDATA DOWNLOADED WITH download_AURNgovdata.py AND WILL SAVE EVERY STATION AS A CSV FILE.
### AT THE THE END OF THE SCRIPT, IT WILL DELETE ALL DE RDATA FILES PER STATIONS BUT WILL MANTAIN THE AURN_metadata FILE IN CASE IS NEEDED.

import os
import shutil
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd

pandas2ri.activate()


def read_rdata_file(file_path, object_name):
    # Reading a specific object from an RData file and return it as a pandas DataFrame.
    ro.r['load'](file_path)
    objects = ro.r.ls()
    if object_name not in objects:
        raise ValueError(f"Object '{object_name}' not found in {file_path}")
    r_object = ro.r[object_name]
    return pandas2ri.rpy2py(r_object)

def convert_unix_to_datetime(df, date_column='date'):
    # Converting Unix timestamps in the DataFrame column to datetime.
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], unit='s')
    return df

def concatenate_dataframes(dataframes):
    # Concatenatenating DataFrames.
    return pd.concat(dataframes, ignore_index=True)

def process_folders(directory, stations_dict, years):
    # Processing folders to extract DataFrames from RData.
    all_dataframes = {}

    for folder, stations in stations_dict.items():
        folder_path = os.path.join(directory, folder)

        if not os.path.isdir(folder_path):
            print(f"Skipping non-directory {folder_path}")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.RData'):
                # Removing extension
                base_name = os.path.splitext(file_name)[0]
                if '_' in base_name:
                    station_name, year = base_name.rsplit('_', 1)
                    if station_name in stations and year.isdigit() and int(year) in years:
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            df = read_rdata_file(file_path, base_name)
                            if isinstance(df, pd.DataFrame):
                                # Converting date column to datetime
                                df = convert_unix_to_datetime(df)
                                if station_name not in all_dataframes:
                                    all_dataframes[station_name] = []
                                all_dataframes[station_name].append(df)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

    return all_dataframes

def concatenate_station_dataframes(dataframes_dict):
    # Concatenatenating DataFrames
    concatenated_dataframes = {}

    for station, dfs in dataframes_dict.items():
        try:
            concatenated_dataframes[station] = concatenate_dataframes(dfs)
            print(f"DataFrames for station '{station}' concatenated.")
        except ValueError as e:
            print(f"Error for station '{station}': {e}")

    return concatenated_dataframes



# Defining the directory path, stations, and years.

directory = 'datasets/AURN_data/'
stations_dict = {
    "Manchester": ["MAN3", "MAHG"],
    "Salford": ["ECCL", "GLAZ"],
    "Bury": ["BURW"]
}
years = [2016, 2017,2018,2019,2020,2021,2022,2023]

# Processing and Concatenating DataFrames
try:
    all_dataframes = process_folders(directory, stations_dict, years)
    concatenated_dataframes = concatenate_station_dataframes(all_dataframes)

    for station, df in concatenated_dataframes.items():
        print(f"Concatenated DataFrame for '{station}' loaded with {len(df)} rows.")

except ValueError as e:
    print(e)

# Cleaning nulls on Pollutants part of the ERP, creating NOx (NO + NO2), and saving results (csv) per each station.

# Manchester Piccadilly (MAN3)
man_picc = concatenated_dataframes["MAN3"]
man_picc["date"] = pd.to_datetime(man_picc["date"], errors = "raise")
man_picc.sort_values(by = "date", inplace = True)
man_picc.reset_index(drop = True, inplace = True)
man_picc["NOx"] = man_picc["NO"] + man_picc["NO2"]
man_picc = man_picc.dropna(subset=['O3', 'NOx', 'PM2.5','wd','ws','temp'])
man_picc.to_csv("datasets/AURN_data/man_picc_AURN.csv",index=False )

# Manchester Sharston (MAHG).
# PM2.5 is not available in this station
man_shar = concatenated_dataframes["MAHG"]
man_shar["date"] = pd.to_datetime(man_shar["date"], errors = "raise")
man_shar.sort_values(by = "date", inplace = True)
man_shar.reset_index(drop = True, inplace = True)
man_shar["NOx"] = man_shar["NO"] + man_shar["NO2"]
man_shar = man_shar.dropna(subset=['O3', 'NOx','wd','ws','temp'])
man_shar.to_csv("datasets/AURN_data/man_shar_AURN.csv",index=False )

# Salford Eccles (ECCL).
# O3 is not availabe in this station as they start taking measures in 2023.
sal_eccl = concatenated_dataframes["ECCL"]
sal_eccl["date"] = pd.to_datetime(sal_eccl["date"], errors = "raise")
sal_eccl.sort_values(by = "date", inplace = True)
sal_eccl.reset_index(drop = True, inplace = True)
sal_eccl["NOx"] = sal_eccl["NO"] + sal_eccl["NO2"]
sal_eccl = sal_eccl.dropna(subset=['PM2.5','NOx','wd','ws','temp'])
sal_eccl.to_csv("datasets/AURN_data/sal_eccl_AURN.csv",index=False )

# Salford Glazebury (GLAZ).
# PM2.5 have a lot of missing data. Not available.
sal_glaz = concatenated_dataframes["GLAZ"]
sal_glaz["date"] = pd.to_datetime(sal_glaz["date"], errors = "raise")
sal_glaz.sort_values(by = "date", inplace = True)
sal_glaz.reset_index(drop = True, inplace = True)
sal_glaz["NOx"] = sal_glaz["NO"] + sal_glaz["NO2"]
sal_glaz = sal_glaz.dropna(subset=['O3', 'NOx','wd','ws','temp'])
sal_glaz.to_csv("datasets/AURN_data/sal_glaz_AURN.csv",index=False )


# Bury Whitefield (BURW).
# PM2.5 and O3 is not available in this station

bury_whit = concatenated_dataframes["BURW"]
bury_whit["date"] = pd.to_datetime(bury_whit["date"], errors = "raise")
bury_whit.sort_values(by = "date", inplace = True)
bury_whit.reset_index(drop = True, inplace = True)
bury_whit["NOx"] = bury_whit["NO"] + bury_whit["NO2"]
bury_whit = bury_whit.dropna(subset=['NOx','wd','ws','temp'])
bury_whit.to_csv("datasets/AURN_data/bury_whit_AURN.csv",index=False )



# Deleting all RData files per station.

# BE CAREFUL HERE!!!! This operation is irreversible; once the directory is deleted, it cannot be recovered.

# Paths to the directories to delete
directories_to_delete = [
    'datasets/AURN_data/Manchester',
    'datasets/AURN_data/Bury',
    'datasets/AURN_data/Salford',
]

for directory_path in directories_to_delete:
    try:
        shutil.rmtree(directory_path)
        print(f"Directory {directory_path} and all its contents have been deleted.")
    except FileNotFoundError:
        print(f"Directory {directory_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: {directory_path}.")
    except Exception as e:
        print(f"Error: {e}")
