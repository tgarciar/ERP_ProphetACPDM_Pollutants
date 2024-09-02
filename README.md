# ERP: Prophet ACPDM for detecting changes in Pollutants Behaviour
An extended research project (ERP) report submitted to the University of Manchester.

- Title: Assessing Prophet automated change point detection method ability to recognise NOx, PM2.5 and O3 fluctuations produced by COVID-19 lockdown policies

- Ob1: Evaluate Prophet's ACPDM proficiency to recognise changes in the behaviour of NOx, PM2.5, and O3 pollutants in Manchester. due to UK lockdown policies during 2020-2021.

- Ob2: Compare various techniques for change point detection within Prophet's ACPDM and provide recommendations for their application in air quality monitoring.

- Ob3: Examine different strategies for weather normalisation of air quality data and their impact on trend change point analyses.

## Table of Contents

- [About this Repository](#about)
- [Installation](#installation)
- [How to use the Repository](#howto)
- [License](#license)

## About this Repository:
This repository aims to make the results and work done for the Extended Research Project reproducible. For this, the code was cleaned, annotated, organized, and automated to follow the main parts of each process. The Technical Appendix accompanies this repository. It explains more about the project and the methods used, which is sure to enhance understanding.

## Installation

1. Clone the repository:
  ```bash
   git clone https://github.com/tgarciar/ERP_ProphetACPDM_Pollutants
   cd ERP_ProphetACPDM_Pollutants
  ```
2.  Install PYTHON==3.10.6
  ```bash
    pyenv install 3.10.6
  ```
3. Set python version
  ```bash
    pyenv local 3.10.6
  ```
4.  Create a Virtual Env. / Activate
  ```bash
    python -m venv venv
    source venv/bin/activate
  ```
5. Install requirements
  ```bash
    pip install -r requirements.txt
  ```
6. Download all big csv files:
  ```bash
    python loading_results.py
  ```

If 6. does not work, you can find the files here:
https://drive.google.com/drive/folders/1O1B5Qol5yRMfGUk5N37GZYcM6GvuZTjy?usp=sharing

The paths where each file needs to be saved are:

forecast_experiment1 --> "experiment1/saved_results/datasets/forecasts_experiment1.csv"
forecast_experiment2 --> "experiment2/saved_results/datasets/forecasts_experiment2.csv"
forecast_experiment3 --> "experiment3/saved_results/datasets/forecasts_experiment3.csv"
ERP_WETNOR_combined_ts --> "weather_normalisation/saved_results/WETNOR_timeseries/ERP_WETNOR_combined_ts.csv"

## How to use the Repository
This repository works in two depths of complexity. The first level corresponds to the Python Scripts found in the main Directory. You might recognise them because they have a number that conforms to the part of the process. These run other scripts inside the folders.

For example, running 0_dataretrieval_script.py initiates two scripts located inside the data_retrieval folder: 01_download_AURNgovdata.py and 02_merging_stations_AURNgovdata.py. This also provides the opportunity to explore 01_download_AURNgovdata.py (second-level deep) and understand its functionality. At this level, you have the flexibility to adjust hyperparameters and examine the code in detail.

The recommendation is to install everything and run in order:

```bash
  python 0_dataretrieval_script.py
```
``` bash
python 1_EDA_results.py
```
    ...

``` bash
python 5_experiment3.py
```

This first approach will generate results inside the folders that the users can start to inspect and understand. The Technical Appendix and the code (annotations) contain more information on the process.

## Licenses

All Licences can be found in the folder --> Licenses/
