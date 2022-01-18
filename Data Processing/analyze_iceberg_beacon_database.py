#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:22:36 2020
Modified on Fri Dec 17 15:00:00 2021

@author: Adam Garbo

analyze_iceberg_beacon_database.py

Description:
    - Used to assess the quality of raw iceberg trajectories contained in the 
    iceberg beacon database and also the suitability of data for use in the 
    validation study of the NAIS iceberg drift model.
    - Calculates speed, distance, and direction of sucessive iceberg positions 
    and visualizes the data in map/graph format.
    
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import seaborn as sns

# -----------------------------------------------------------------------------
# Library Configuration
# -----------------------------------------------------------------------------

# Add Natural Earth coastline
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="black", facecolor="lightgray", lw=0.75
)

# Add Natural Earth coastline
coastline = cfeature.NaturalEarthFeature(
    "physical", "coastline", "10m", edgecolor="black", facecolor="none", lw=0.75
)

# Configure Seaborn styles
sns.set_theme(style="ticks")
sns.set_context("talk")  # Options: talk, paper, poster

# Configure legend outline (optional)
# plt.rc("legend", fancybox=False, framealpha=1, edgecolor="k")

# Set colour palette
colour = ["red", "lime", "blue", "magenta", "cyan", "yellow"]
sns.set_palette(colour)

# Set figure DPI
dpi = 500

# -----------------------------------------------------------------------------
# Option 1) Create individual maps for each iceberg contained in the iceberg  
# tracking beacon database 
# Requires: Compiled CSV file of the iceberg beacon database
# -----------------------------------------------------------------------------

# Load most recent iceberg beacon database CSV file
df = pd.read_csv(
    "/Volumes/data/cis_iceberg_beacon_database/output_data/csv/database_20210622.csv",
    index_col=False,
)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

# Load beacon IDs that will be used to subset the database
subset = pd.read_csv(
    "/Users/adam/My Drive/University of Ottawa/Thesis/NAIS Iceberg Drift Model/input/subset.csv"
)

# Change column names to lowercase
subset.columns = subset.columns.str.lower()

# Subset the database
df = df[df.set_index(["beacon_id"]).index.isin(subset.set_index(["beacon_id"]).index)]

# Loop to plot each beacon id
for label, group in df.groupby(["beacon_type", "beacon_id"]):

    # Calculate the length of the iceberg track
    duration = (group["datetime_data"].max() - group["datetime_data"].min()).days

    # Calculate cumulative distance of the iceberg track
    distance = group["distance"].sum() / 1000

    plt.figure(figsize=(12, 12))
    ax = plt.axes(
        projection=ccrs.Orthographic(
            ((group["longitude"].min() + group["longitude"].max()) / 2),
            (group["latitude"].min() + group["latitude"].max()) / 2,
        )
    )
    ax.add_feature(coast)
    
    #ax.gridlines(draw_labels=True, color="black", alpha=0.5, linestyle="-")
    gl = ax.gridlines(
        draw_labels=True,
        color="black",
        alpha=0.5,
        linestyle="dotted",
    )
    gl.top_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.right_labels = False
    gl.rotate_labels = False
    sns.lineplot(
        x="longitude",
        y="latitude",
        data=group,
        color="red",
        lw=2,
        ci=None,
        sort=False,
        estimator=None, 
        transform=ccrs.PlateCarree(),
        ax=ax,
    )
    # Make maps square
    ax.set_adjustable("datalim")
    plt.title(
        "%s %s\n%s to %s\n%s days %.2f km"
        % (
            label[0],
            label[1],
            group["datetime_data"].min(),
            group["datetime_data"].max(),
            duration,
            distance.sum(),
        ),
        loc="left",
    )

    # Save figure
    plt.savefig(
        path_figures + "maps/%s.png" % label[1],
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
    )
    
    break
    # Close plot
    plt.close()

# -----------------------------------------------------------------------------
# Option 2) Plot trajectory analysis for each iceberg contained in the iceberg  
# tracking beacon database 
# Requires: Compiled CSV file of the iceberg beacon database
# -----------------------------------------------------------------------------

# Load most recent iceberg beacon database CSV file
df = pd.read_csv(
    "/Volumes/data/cis_iceberg_beacon_database/release/csv/database_20210622.csv",
    index_col=False,
)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

# Load beacon IDs that will be used to subset the database
subset = pd.read_csv(
    "/Users/adam/My Drive/University of Ottawa/Thesis/NAIS Iceberg Drift Model/input/subset.csv"
)

# Change column names to lowercase
subset.columns = subset.columns.str.lower()

# Subset the database
df = df[df.set_index(["beacon_id"]).index.isin(subset.set_index(["beacon_id"]).index)]


# Initialize pyproj with appropriate ellipsoid
geodesic = pyproj.Geod(ellps="WGS84")
    
for label, group in df.groupby(["beacon_type", "beacon_id"]):

    # Calculate forward azimuth and Great Circle distance between successive beacon positions
    group["azimuth_obs"], back_azimuth, group["distance"] = geodesic.inv(
        group["longitude"].shift().tolist(),
        group["latitude"].shift().tolist(),
        group["longitude"].tolist(),
        group["latitude"].tolist(),
    )

    # Convert azimuth from (-180° to 180°) to (0° to 360°)
    group["azimuth_obs"] = (group["azimuth_obs"] + 360) % 360

    # Convert distance to kilometres
    group["distance"] = group["distance"] / 1000.0

    # Calculate speed
    group["speed_ms"] = (group["distance"] * 1000) / group[
        "datetime_data"
    ].diff().dt.seconds

    # Calculate the length of the iceberg track
    duration = (group["datetime_data"].max() - group["datetime_data"].min()).days

    # Calculate cumulative distance of the iceberg track
    distance = group["distance"].sum()

    # Create figure and add axes object
    fig = plt.figure(figsize=(15, 10))

    # Add axes (size rows, columns), (location rows, columns)
    ax = plt.subplot2grid(
        (3, 2),
        (0, 0),
        rowspan=3,
        colspan=1,
        projection=ccrs.Orthographic(
            group["longitude"].median(), group["latitude"].median()
        ),
    )
    ax.add_feature(coast)
    gl = ax.gridlines(
        draw_labels=True,
        color="black",
        alpha=0.5,
        linestyle="dotted",
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.right_labels = False
    gl.rotate_labels = False

    # Lineplot of observed iceber trajectories
    sns.lineplot(
        x="longitude",
        y="latitude",
        color="red",
        data=group,
        ci=None,
        sort=False,
        estimator=None,
        transform=ccrs.PlateCarree(),
        ax=ax,
    )
    ax.set_adjustable("datalim")

    plt.title(
        "%s %s\n%s to %s\n%s days %.2f km"
        % (
            label[0],
            label[1],
            group["datetime_data"].min(),
            group["datetime_data"].max(),
            duration,
            distance,
        ),
        loc="left",
    )

    # Line graph 1
    ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
    sns.lineplot(
        x="datetime_data", y="speed_ms", data=group, color="blue", lw=1, ci=None, ax=ax1
    )
    sns.despine()
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(alpha=0.5, linestyle="dashed")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax1.set(xticklabels=[], xlabel=None)

    # Line graph 2
    ax2 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1)
    sns.lineplot(
        x="datetime_data",
        y="distance",
        data=group,
        color="green",
        lw=1,
        ci=None,
        ax=ax2,
    )
    sns.despine()
    ax2.set_ylabel("Distance (km)")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.set(xticklabels=[], xlabel=None)

    # Line graph 3
    ax3 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
    sns.lineplot(
        x="datetime_data", y="ta", data=group, color="orange", lw=1, ci=None, ax=ax3
    )
    sns.lineplot(
        x="datetime_data", y="ti", data=group, color="purple", lw=1, ci=None, ax=ax3
    )
    sns.lineplot(
        x="datetime_data", y="ts", data=group, color="cyan", lw=1, ci=None, ax=ax3
    )
    sns.despine()
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(alpha=0.5, linestyle="dashed")
    ax3.set(xlabel=None)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)

    # Save figure
    plt.savefig(
        path_figures + "%s.png" % label[1],
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
    )

    # Close plot
    plt.close()

    # Debug: Include break to iterate through the loop only once
    break


# ----------------------------------------------------------------------------
# Option 3) Plot trajectory analysis for each iceberg selected for use in the
# validation study of the NAIS iceberg drift model
# Requires: A folder of individual raw iceberg tracjectories extracted from the 
# iceberg tracking beacon database
# -----------------------------------------------------------------------------

# Path to data
path_input = "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/raw/"
path_figures = "/Volumes/data/nais_iceberg_drift_model/figures/trajectory_analysis/"

# Find all files in folder
files = glob.glob(path_input + "*.csv")

# Iterate through each raw iceberg tracjetory and produce analysis figures
for file in files:

    # Debug: Print filename being processed
    print(file)

    # Plot analysis figures
    plot_analysis(file)
    
    # Debug: Stop loop after one iteration
    break


def plot_analysis(filename):
    """

    Calculate the distance, speed and direction between successive coordinates.

    Produce a four panel figure, including:
        1) Trajectory map
        2) Speed (m/s))
        3) Distance (km)
        4) Temperature:
             Ta = Air temperature 
             Ti = Internal temperature
             Ts = Sea surface temperature

    Parameters
    ----------
    filename : str
        Input CSV of raw iceberg tracjectory extracted from the iceberg 
        tracking beacon database.

    Returns
    -------
    None.

    """

    # Initialize pyproj with appropriate ellipsoid
    geodesic = pyproj.Geod(ellps="WGS84")

    # Splice beacon ID
    label = os.path.splitext(os.path.basename(filename))[0]

    # Load selected beacon IDs that will be used to subset the database
    df = pd.read_csv(filename, index_col=False)

    # Convert to datetime
    df["datetime_data"] = pd.to_datetime(
        df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )

    # Calculate forward azimuth and Great Circle distance between successive beacon positions
    df["azimuth_obs"], back_azimuth, df["distance"] = geodesic.inv(
        df["longitude"].shift().tolist(),
        df["latitude"].shift().tolist(),
        df["longitude"].tolist(),
        df["latitude"].tolist(),
    )

    # Convert azimuth from (-180° to 180°) to (0° to 360°)
    df["azimuth_obs"] = (df["azimuth_obs"] + 360) % 360

    # Convert distance to kilometres
    df["distance"] = df["distance"] / 1000.0

    # Calculate speed
    df["speed_ms"] = (df["distance"] * 1000) / df["datetime_data"].diff().dt.seconds

    # Calculate the length of the iceberg track
    duration = df["datetime_data"].max() - df["datetime_data"].min()

    # Calculate cumulative distance of the iceberg track
    distance = df["distance"].sum()

    # Create figure and add axes object
    fig = plt.figure(figsize=(17.5, 10))

    # Add axes (size rows, columns), (location rows, columns)
    ax = plt.subplot2grid(
        (3, 2),
        (0, 0),
        rowspan=3,
        colspan=1,
        projection=ccrs.Orthographic(
            ((df["longitude"].min() + df["longitude"].max()) / 2),
            (df["latitude"].min() + df["latitude"].max()) / 2,
        )
    )
    ax.add_feature(coast)
    gl = ax.gridlines(
        draw_labels=True,
        color="black",
        alpha=0.5,
        linestyle="dotted",
    )
    gl.top_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.right_labels = False
    gl.rotate_labels = False

    # Lineplot of observed iceberg trajectories
    sns.lineplot(
        x="longitude",
        y="latitude",
        data=df,
        color="red",
        lw=2,
        ci=None,
        sort=False,
        estimator=None, 
        transform=ccrs.PlateCarree(),
        ax=ax,
    )

    # Make maps square
    ax.set_adjustable("datalim")

    # Set title
    plt.title(
        "Beacon: %s\nStart: %s End: %s\nDuration: %s Distance: %.2f km"
        % (
            label,
            df["datetime_data"].min(),
            df["datetime_data"].max(),
            duration,
            distance,
        ),
        loc="left",
    )

    # Plot speed
    ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
    sns.lineplot(
        x="datetime_data", y="speed_ms", data=df, color="blue", lw=1, ci=None, ax=ax1
    )
    sns.despine()
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(alpha=0.5, linestyle="dashed")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax1.set(xticklabels=[], xlabel=None)

    # Plot Distance
    ax2 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, sharex=ax1)
    sns.lineplot(
        x="datetime_data", y="distance", data=df, color="green", lw=1, ci=None, ax=ax2
    )
    sns.despine()
    ax2.set_ylabel("Distance (km)")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.set(xticklabels=[], xlabel=None)

    # Plot temperature
    ax3 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1, sharex=ax1)
    sns.lineplot(
        x="datetime_data", y="ta", data=df, color="orange", lw=1, ci=None, ax=ax3
    )
    sns.lineplot(
        x="datetime_data", y="ti", data=df, color="purple", lw=1, ci=None, ax=ax3
    )
    sns.lineplot(
        x="datetime_data", y="ts", data=df, color="cyan", lw=1, ci=None, ax=ax3
    )
    sns.despine()
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(alpha=0.5, linestyle="dashed")
    ax3.set(xlabel=None)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    fig.autofmt_xdate(rotation=45)

    # Save figure
    fig.savefig(
        path_figures + "%s.png" % label,
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
    )

    # Close plot
    plt.close(fig)