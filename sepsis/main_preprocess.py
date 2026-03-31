# TODO
# Add a optional_analysis function post training

__author__ = 'Aaron J Masino'

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sepsis import imputation, log_worker

# *******************************  GLOBAL PARAMETERS ******************************************
# files & directories
date_string='2018-05-14'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
raw_dir = os.path.join(data_dir, "raw")
interim_dir = os.path.join(data_dir, "interim")
processed_dir = os.path.join(data_dir, "processed")
os.makedirs(interim_dir, exist_ok=True)

case_file = os.path.join(raw_dir, "CASES_FILE.csv")
control_file = os.path.join(raw_dir, "CONTROLS_FILE.csv")

# limit of missing data percentage for imputation. Data missing at rate above cutoff will be
# dropped from data (enter as integer percent)
missing_cutoff = 80

# set to [] if not dropping any columns - columns may me dropped based on missing percentage
cols_to_drop = []

# column names for features requiring normalization
cols_to_norm = ['gest_age', 'age', 'wbc', 'hgb', 'it_ratio',
       'capPH', 'bicarb', 'glucose', 'creatinine', 'platelet_count', 'hr',
       'rr', 'temp', 'sbp', 'dbp', 'map', 'weight', 'fio2',
       'hr_delta', 'rr_delta', 'mabp_delta',
       'temp_delta']


# Sepsis Groups
# Group 1: Culture Positive sepsis: culture positive, minimum 5 days antibiotic treatment
# Group 2: Negative Sepsis: Culture negative, <72 hours antibiotic treatment
# Group 3: Clinical Sepsis: Culture negative, >120 hours of antibiotic treatment

# sepsis groups to use in positive samples (cases):
CASE_GRPS = [1,3]

# control group data is from non-septic periods for same individuals in case groups
# if using controls file with unspecified groups, set to None
CONTROL_GRPS = None

file_prefix='temp_'
# ************************************ GLOBAL PARAMETERS END *************************************

# import data
cases = pd.read_csv(case_file)
controls = pd.read_csv(control_file)
cases['sepsis'] = 1
controls['sepsis'] = 0

# select data from specified sepsis groups
if CONTROL_GRPS is None:
    controls['sepsis_group'] = -1
    CONTROL_GRPS = [-1]
X = pd.concat([cases[cases['sepsis_group'].isin(CASE_GRPS)],
               controls[controls['sepsis_group'].isin(CONTROL_GRPS)]])

# log case / control stats
of = os.path.join(interim_dir, "case_control_stats.txt")
log_worker.log_line("samples count: {0}\n".format(len(X)), of)
y = X['sepsis']
log_worker.log_line("target counts\n{0}\n".format(y.value_counts()), of)
log_worker.log_line("Incidence Rate (percent): {0:.3f}\n".format(100 * np.sum(y) / float(len(y))), of)

# Imputation
missing_percentages = imputation.missing_percents(X)
log_worker.log_dictionary(missing_percentages, "Missing Data Percentages\n", False,
                          os.path.join(interim_dir, "missing_data_percents.txt"))

# drop columns with missing percentage over threshold
#cols_to_drop = []
for k,v in missing_percentages.items():
    if v > missing_cutoff:
        cols_to_drop.append(k)

for c in cols_to_drop:
    X = X.drop(c,axis=1)
    if c in cols_to_norm:
        cols_to_norm.remove(c)

# impute missing values
imp = SimpleImputer(strategy='mean')
Ximp = imp.fit_transform(X)
cnames = X.columns
X = pd.DataFrame(data=Ximp, columns=cnames)

# Normalization
X[cols_to_norm] = X[cols_to_norm].apply(lambda x: (x-x.mean())/x.std())

## Store processed data
X.to_csv(os.path.join(processed_dir, "preprocessed_data.csv"), index=False)
