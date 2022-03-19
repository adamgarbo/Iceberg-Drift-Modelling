#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 4 10:36:46 2021

@author: Adam Garbo

create_iceberg_input_file.py

Description:
    - Creates input file of icebergs to be modelled by the NAIS iceberg drift model.
    - Reads the most recent version of the iceberg trackign beacon database subset and 
    extracts the positions and times of of every 96-hour window.
    - Reads the iceberg dimensions file and assigns these values to data 
    contained in each 96-hour period.
    - Outputs input variables required by the NAIS model (typically included in Berg.in).
"""

import pandas as pd

# Load observed iceberg trajectories
df1 = pd.read_csv(
    "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/merged/interpolated.csv",
    usecols=["beacon_id", "datetime_data", "longitude", "latitude"],
    index_col=False,
)

# Load iceberg dimensions
df2 = pd.read_csv(
    "/Volumes/data/nais_iceberg_drift_model/input/subset.csv", index_col=False
)

# Clear interval column
df2["interval"] = 0

# Merge dataframes
df3 = pd.merge(df1, df2, how="left", on="beacon_id")

# Convert to datetime
df3["datetime_data"] = pd.to_datetime(
    df3["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

# Create an empty dataframe
df4 = pd.DataFrame(
    columns=["beacon_id", "datetime_data", "longitude", "latitude", "interval"]
)

# Group data according to beacon ID
for label, group in df3.groupby("beacon_id"):

    # Interval counter
    i = 0

    # Set interval of first row of each group to 0
    group["interval"].iloc[0] = 0

    # Append first row of each group (i.e. start time)
    df4 = df4.append(group.iloc[0])

    # Record start time
    start_time = group["datetime_data"].iloc[0]

    # Calculate desired interval
    interval = start_time + pd.Timedelta(hours=96)

    # Iterate through all rows of each group
    for index, row in group.iterrows():

        # Append rows with a delta equal to or greater than interval
        if row["datetime_data"] >= interval:
            i += 1
            row["interval"] = i
            df4 = df4.append(row)
            # Update rolling interval
            interval = row["datetime_data"] + pd.Timedelta(hours=96)

# Reset index
df4 = df4.reset_index(drop=True)

# Create dataframe according to column structure used by NAIS model Berg.in file
icebergs = pd.DataFrame(
    columns=[
        "target",
        "id",
        "branch",
        "start_time",
        "latitude",
        "longitude",
        "size",
        "shape",
        "mobility",
        "length",
        "melt",
        "speed",
        "direction",
        "sail",
        "keel",
    ]
)

# Assign values to dataframe
icebergs["id"] = df4["beacon_id"]
icebergs["target"] = "SNGL"
# icebergs['id'] = range(1, len(icebergs)+1)
icebergs["branch"] = df4["interval"]
icebergs["start_time"] = df4["datetime_data"]
icebergs["latitude"] = df4["latitude"]
icebergs["longitude"] = df4["longitude"]
icebergs["size"] = "LRG"  # Not considered by model
icebergs["shape"] = df4["shape"]
icebergs["mobility"] = "DFT"  # Not considered by model
icebergs["length"] = df4["length"]
icebergs["melt"] = 0
icebergs["speed"] = 0
icebergs["direction"] = 0
icebergs["sail"] = df4["freeboard"]
icebergs["keel"] = df4["draft"]

# Write icebergs to input file in CSV format
icebergs.to_csv(
    "/Volumes/data/nais_iceberg_drift_model/input/iceberg_input.csv", index=False
)
