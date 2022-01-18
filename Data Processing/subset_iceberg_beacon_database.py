#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:31:03 2021

@author: Adam Garbo

Description:
    - Iteratively searches through the iceberg beacon database and extracts 
    the relevant raw data files.
    - Merges all raw data files into a single raw data file for further
    processing.

"""

import csv
import os
import glob
import shutil
import time
import pandas as pd

# -----------------------------------------------------------------------------
# Extract subset of iceberg tracks from iceberg beacon database
# -----------------------------------------------------------------------------

# Path to observed tracks
path_input = "/Volumes/data/cis_iceberg_beacon_database/data/"
path_output = "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/raw/"

# Load subset of iceberg beacon database
df = pd.read_csv("/Volumes/data/nais_iceberg_drift_model/input/subset.csv")

# Append .csv file extension
df["BEACON_ID"] = df["BEACON_ID"] + ".csv"

# Create set from the beacon IDs
subset = df["BEACON_ID"].unique()

# Recursively search through all files to locate
files = sorted(
    glob.glob(path_input + "/**/standardized_data/*.csv", recursive=True)
)

# Copy all files to output path
for file in files:
    # print(os.path.basename(file))
    if os.path.basename(file) in subset:
        shutil.copy(file, path_output)  # Copy files to new directory


# -----------------------------------------------------------------------------
# Merge files
# -----------------------------------------------------------------------------

datetime = time.strftime("%Y%m%d")
filename = "output" + datetime + ".csv"

# Change accordingly
input_path = "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/raw"  
output_path = "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/merged/raw.csv"  # + filename
files = sorted(glob.glob(input_path + "/*.csv"))
print(files)

# Concatenate CSV files
with open(output_path, "w") as outfile:
    for i, file in enumerate(files):
        with open(file, "r") as infile:
            if i != 0:
                infile.readline()  # Throw away header on all but first file
            # Block copy rest of file from input to output without parsing
            shutil.copyfileobj(infile, outfile)
            print(file + " has been imported.")
