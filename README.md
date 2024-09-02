# ERP: Prophet ACPDM for detecting changes in Pollutants Behaviour
An extended research project (ERP) report submitted to the University of Manchester.

- Title: Assessing Prophet automated change point detection method ability to recognise NOx, PM2.5 and O3 fluctuations produced by COVID-19 lockdown policies

- Ob1: Evaluate Prophet's ACPDM proficiency to recognise changes in the behaviour of NOx, PM2.5, and O3 pollutants in Manchester. due to UK lockdown policies during 2020-2021.

- Ob2: Compare various techniques for change point detection within Prophet's ACPDM and provide recommendations for their application in air quality monitoring.

- Ob3: Examine different strategies for weather normalisation of air quality data and their impact on trend change point analyses.

## Table of Contents

- [About this Repository](#about)
- [How to use the Repo.](#about)
- [Installation](#installation)
- [License](#license)

## About this Repository:
This repositor







## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tgarciar/ERP_ProphetACPDM_Pollutants
   cd ERP_ProphetACPDM_Pollutants

2. Install PYTHON==3.10.6
   ```bash
    pyenv install 3.10.6

3. Set python version local
   ```bash
    pyenv local 3.10.6

4. Create a Virtual Env. / Activate
   ```bash
    python -m venv venv
    source venv/bin/activate

5. Install requirements
   ```bash
    pip install -r requirements.txt

6. Download all big csv files:
  ```bash
    python loading_results.py

------------------------------------------------
If 6. does not work, you can find the files here:
https://drive.google.com/drive/folders/1O1B5Qol5yRMfGUk5N37GZYcM6GvuZTjy?usp=sharing

The paths where each file needs to be saved are:

forecast_experiment1 --> "experiment1/saved_results/datasets/forecasts_experiment1.csv"
forecast_experiment2 --> "experiment2/saved_results/datasets/forecasts_experiment2.csv"
forecast_experiment3 --> "experiment3/saved_results/datasets/forecasts_experiment3.csv"
ERP_WETNOR_combined_ts --> "weather_normalisation/saved_results/WETNOR_timeseries/ERP_WETNOR_combined_ts.csv"

## Licenses

All Licences can be found in the folder --> Licenses/
