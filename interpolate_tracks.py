#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:01:32 2021

@author: Adam Garbo

interpolate_tracks.py

Description:
   Interpolates irregular iceberg observations to a standardized time interval 
   between observations.

"""

import os
import glob
import shutil
import pandas as pd

# ----------------------------------------------------------------------------
# Batch perform interpolations
# ----------------------------------------------------------------------------

# Path to cleaned model input files
path_input = "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/cleaned/"

# Path of interpolated ouput files
path_output = (
    "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/interpolated/"
)

# Find all files in folder
files = glob.glob(path_input + "/*.csv")

# Iterate through each error file and create plots
for file in files:

    print(file)
    # Plot error
    interpolate_tracks(file)

# ----------------------------------------------------------------------------
# Interpolation Function
# ----------------------------------------------------------------------------


def interpolate_tracks(filename):
    """


    Parameters
    ----------
    filename : str
        Path to CSV of cleaned iceberg tracjectory extracted from the iceberg
        tracking beacon database.

    Returns
    -------
    None.

    """

    # Get filename
    fname = os.path.splitext(os.path.basename(file))[0]

    # Read input file
    df = pd.read_csv(
        filename, usecols=["datetime_data", "longitude", "latitude"], index_col=False
    )

    # Set datetime
    df["datetime_data"] = pd.to_datetime(
        df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )

    # Optional: Ensure datetime seconds are zeroed to avoid uncessary interpolation
    # df['datetime_data'] = df['datetime_data'].dt.floor('min')

    # Label to track number of performed interpolations
    df["interp"] = "x"

    # Create datetime index to interpolate modelled data
    dti = pd.date_range(
        df["datetime_data"].dt.floor("H").min(),  # Round down hour of first location
        df["datetime_data"].dt.ceil("H").max(),  # Round up hour of last location
        freq="H",
    )

    # Set datatime as index of dataframes for merge
    df = df.set_index("datetime_data")

    # Interpolate data
    df2 = df.reindex(df.index.union(dti)).interpolate(method="index").reindex(dti)

    # Count interpolations
    df2["interpolated"] = (df2["interp"].isnull()).cumsum()

    # Drop temporary column
    df2 = df2.drop(columns=["interp"])

    # Reset index
    df2 = df2.reset_index()

    # Rename column
    df2 = df2.rename(columns={"index": "datetime_data"})

    # Drop NaN rows
    df2 = df2.dropna()

    # Calculate interval between observations
    df2["duration"] = (df2["datetime_data"].diff().dt.seconds / 3600.0).cumsum()

    # Reassign unique identifier
    df2["beacon_id"] = fname

    # Save to CSV file
    df2.to_csv(path_output + "%s.csv" % fname, index=False)


# ----------------------------------------------------------------------------
# Merge error metrics into a single CSV file
# ----------------------------------------------------------------------------

path_input = (
    "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/interpolated/"
)
path_output = "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/merged/interpolated.csv"
files = sorted(glob.glob(path_input + "*.csv"))

# Concatenate CSV files
with open(path_output, "w") as outfile:
    for i, file in enumerate(files):
        with open(file, "r") as infile:
            if i != 0:
                infile.readline()  # Throw away header on all but first file
            # Block copy rest of file from input to output without parsing
            shutil.copyfileobj(infile, outfile)
            print(file + " has been imported.")
