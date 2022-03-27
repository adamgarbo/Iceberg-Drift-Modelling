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


import scipy
import pandas

# -----------------------------------------------------------------------------
# Library Configuration
# -----------------------------------------------------------------------------

# Path to data
path_data = ""

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
df = load_data(2090)
