# ERP: Prophet ACPDM for detecting changes in Pollutants Behaviour
An extended research project (ERP) report submitted to the University of Manchester.

Title: Assessing Prophet automated change point detection method ability to recognise NOx, PM2.5 and O3 fluctuations produced by COVID-19 lockdown policies

- Ob1: Evaluate Prophet's ACPDM proficiency to recognise changes in the behaviour of NOx, PM2.5, and O3 pollutants in Manchester. due to UK lockdown policies during 2020-2021.

- Ob2: Compare various techniques for change point detection within Prophet's ACPDM and provide recommendations for their application in air quality monitoring.

- Ob3: Examine different strategies for weather normalisation of air quality data and their impact on trend change point analyses.

## Table of Contents

- [About this Repository](#about)
- [Installation](#installation)
- [Usage](#usage)
  - [Weather Normalisation](#weather-normalisation)
  - [Experiment 1](#experiment-1)
- [Parameters](#parameters)
- [Contributing](#contributing)
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

6. Download all the big csv files:
  ```bash
  python loading_results.py
