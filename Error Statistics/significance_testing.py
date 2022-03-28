#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:42:38 2022

Description: 
    - Python code to test for significance of datasets used in the thesis: 
    "Validation of the North American Ice Service Iceberg Drift Model"
    
Notes:
    - Python code formatted using Black:
    https://github.com/psf/black
    
"""


from scipy import stats
import pandas as pd

# -----------------------------------------------------------------------------
# Library Configuration
# -----------------------------------------------------------------------------

# Path to data
path_data = "/Volumes/GoogleDrive/My Drive/University of Ottawa/Thesis/NAIS Iceberg Drift Model/"

/Volumes/GoogleDrive/My Drive/University of Ottawa/Thesis/NAIS Iceberg Drift Model/output/validation/merged/error_2009.csv

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------


def load_data(year):
    """


    Parameters
    ----------
    year : int
        Starting year of hindcast period (2009 or 2017).

    Returns
    -------
    df : pandas DataFrame
        Returns a pandas DataFrame of the selected hindcast period.

    """

    # Read CSV
    df = pd.read_csv(
        path_data + "output/validation/merged/error_%s.csv" % year,
        index_col=False,
    )

    # Drop nan rows
    df = df.dropna()

    # Convert to datetime
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )

    # Add duration column
    df["dur_obs"] = df["dur_obs"].astype(int)

    # Add ocean current model column
    df["current"] = df["branch"].str.split("_").str[1]

    if year == 2009:
        df = df[df["current"] != "riops"]  # Drop RIOPS

    return df



# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# Load data
df = load_data(2017)

# Subset data
#df = df[df['dur_obs'] == 96] 
# Get final distance error
df = df.groupby("ensemble").tail(1)


df1 = df[df["current"] == "cecom"]
df2 = df[df["current"] == "glorys"]
df3 = df[df["current"] == "riops"]

# Perform the Shapiro-Wilk test for normality.

# Observed
stats.shapiro(df1["dist_csum_obs"])

# CECOM
stats.shapiro(df1["dist_csum_model"])
stats.shapiro(df1["dist_error"])

# GLORYS
stats.shapiro(df2["dist_csum_model"])
stats.shapiro(df2["dist_error"])

# RIOPS
stats.shapiro(df3["dist_csum_model"])
stats.shapiro(df3["dist_error"])

# Compute the Kruskal-Wallis H-test for independent samples

# Cumulative distance error
stats.kruskal(df1["dist_csum_model"], df2["dist_csum_model"])
stats.kruskal(df1["dist_csum_model"], df3["dist_csum_model"])
stats.kruskal(df2["dist_csum_model"], df3["dist_csum_model"])

# Distance error
stats.kruskal(df1["dist_error"], df2["dist_error"])
stats.kruskal(df1["dist_error"], df3["dist_error"])
stats.kruskal(df2["dist_error"], df3["dist_error"])


