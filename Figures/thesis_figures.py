#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last modified: 2022-01-18

@author: Adam Garbo

Description: 
    - Python code to produce figures contained in the thesis: 
    "Validation of the North American Ice Service Iceberg Drift Model"
    
Notes:
    - Python code formatted using Black:
    https://github.com/psf/black
    
"""

# -----------------------------------------------------------------------------
# Load librarires
# -----------------------------------------------------------------------------

import os
import glob
import shutil
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

from shapely.geometry import Point
import string

import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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

# Configure
plt.rc("legend", fancybox=False, framealpha=1, edgecolor="k")

# Set colour palette
colour = ["red", "lime", "blue", "magenta", "cyan", "yellow"]
sns.set_palette(colour)

# Set figure DPI
dpi = 500

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Path to figures
path_figures = "/Users/adam/My Drive/University of Ottawa/Thesis/Figures/"

# Path to data
path_data = "/Volumes/data/nais_iceberg_drift_model/"

# -----------------------------------------------------------------------------
# Statistics Functions
# -----------------------------------------------------------------------------

# Root mean square error (RMSE)
def rmse(values):
    return np.sqrt(sum(values ** 2) / len(values))


# Mean absolute error (MAE)
def mae(values):
    return np.average(np.abs(values))


# Standard deviation
def sd(values):
    return np.std(values)


# Coefficient of variation
def cv(values):
    return np.std(values) / np.mean(values) * 100


# Calculate statistics for Figures: 4.6, 4.10
def calculate_statistics(df):
    # Create an empty dataframe
    stats = pd.DataFrame()
    stats = (
        df.groupby(["dur_obs", "current"])["dist_error"]
        .apply(mae)
        .reset_index(name="mae")
    )
    stats["sd"] = (
        df.groupby(["dur_obs", "current"])["dist_error"]
        .apply(sd)
        .reset_index()["dist_error"]
    )
    stats["cv"] = (
        df.groupby(["dur_obs", "current"])["dist_error"]
        .apply(cv)
        .reset_index()["dist_error"]
    )
    stats["rmse"] = (
        df.groupby(["dur_obs", "current"])["dist_error"]
        .apply(rmse)
        .reset_index()["dist_error"]
    )
    stats["scaled_error"] = (
        df.groupby(["dur_obs", "current"])["scaled_error"]
        .apply(rmse)
        .reset_index()["scaled_error"]
    )
    stats["count"] = (
        df.groupby(["dur_obs", "current"]).size().reset_index(name="count")["count"]
    )
    stats = stats.round({"mae": 2, "rmse": 2, "sd": 2, "cv": 2, "scaled_error": 2})

    return stats


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
# Figure 2.1
# Annual counts and monthly averages of icebergs crossing 48°N from 1900-2020
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load annual counts
df1 = pd.read_csv(
    path_figures + "Data/G10028_Icebergs_South_of_48N.csv", index_col=False
)

# Calcualte monthly averages
df2 = df1.drop(columns=["YEAR", "TOTAL"])
df2 = df2.agg(["mean"])
df2 = pd.melt(df2)
df2.columns = ["month", "mean"]
df2["Month"] = pd.to_datetime(df2.month, format="%b", errors="coerce").dt.month
df2 = df2.sort_values(by="Month")

# Plot
_, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
sns.barplot(x="YEAR", y="TOTAL", data=df1, color="black", ax=axs[0])
axs[0].set_xticklabels(
    axs[0].get_xticklabels(), rotation=45, horizontalalignment="center"
)
axs[0].set(xlabel="Year", ylabel="Iceberg Count")
loc = mticker.MultipleLocator(base=10.0)  # Place ticks at regular intervals
axs[0].xaxis.set_major_locator(loc)
sns.barplot(x="month", y="mean", data=df2, color="grey", edgecolor="black", ax=axs[1])
axs[1].set_xticklabels(
    axs[1].get_xticklabels(), rotation=45, horizontalalignment="center"
)
axs[1].set(xlabel="Month", ylabel="Iceberg Count")
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):  # Add figure annotations
    ax.text(
        -0.15,
        -0.25,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=25,
        weight="bold",
    )

# Save figure
plt.savefig(
    path_figures + "Figure_2.1.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)

# -----------------------------------------------------------------------------
# Figure 3.5
# Duration and monthly distribution of iceberg tracks
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load interpolated iceberg drift observation data
df = pd.read_csv(
    path_data + "/input/iceberg_database/merged/interpolated.csv",
    index_col=False,
)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

# Add columns
df["year"] = df["datetime_data"].dt.year
df["month_int"] = df["datetime_data"].dt.month
df["month_str"] = (
    pd.to_datetime(df["month_int"], format="%m").dt.month_name().str.slice(stop=3)
)
df = df.sort_values(by="month_int")

# Subset
df2 = df.groupby("beacon_id", as_index=False)["duration"].max()
# Create new column of just minutes
df2["day"] = df2["duration"] / 24

# Plot max duration in days
fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
axs[0].set(xlabel="Duration (days)", ylabel="Frequency")
sns.histplot(x="day", data=df2, binwidth=1, color="grey", edgecolor="black", ax=axs[0])
axs[1].set(xlabel="Month", ylabel="Count")
sns.histplot(
    x="month_str", data=df, binwidth=1, color="grey", edgecolor="black", ax=axs[1]
)
sns.despine()
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.175,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
# Save figure
fig.savefig(
    path_figures + "Figure_3.5.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)

# -----------------------------------------------------------------------------
# Figure 3.6
# 7-day rolling mean and standard deviation of sea-surface temperatures used
# to determine when beacon fell into the ocean
# Relevant StackOverflow: https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load raw iceberg tracking beacon data
df = pd.read_csv(
    "/Volumes/data/cis_iceberg_beacon_database/data/2011/300234010959690/raw_data/original_file/300234010959690_2011.csv",
    index_col=False,
)

# Convert to datetime
df["DataDate_UTC"] = pd.to_datetime(
    df["DataDate_UTC"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

# Set datetime as index
df = df.set_index(["DataDate_UTC"])

# Calculate rolling mean and standard deviation
df["mean"] = df["SST"].rolling(7, center=False).mean()
df["sd"] = df["SST"].rolling(7, center=False).std()

# Plot rolling mean
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
sns.lineplot(
    x="DataDate_UTC",
    y="mean",
    data=df.loc["2011-7-30":"2011-10-04"],
    lw=1,
    ci=None,
    color="red",
    ax=axs[0],
)
axs[0].set(xlabel=" ", ylabel="Temperature (°C)")
axs[0].annotate(
    "Transition to water",
    xy=(datetime.datetime(2011, 9, 12, 12, 0), 4.1),
    xycoords="data",
    xytext=(-25, 75),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="black"),
    va="center",
    ha="center",
    fontsize=15,
)
sns.lineplot(
    x="DataDate_UTC",
    y="sd",
    data=df.loc["2011-7-30":"2011-10-04"],
    lw=1,
    ci=None,
    color="blue",
    ax=axs[1],
)
axs[1].set(xlabel="Year", ylabel="Standard Deviation (°C)")
axs[1].annotate(
    "Transition to water",
    xy=(datetime.datetime(2011, 9, 12, 12, 0), 0.1),
    xycoords="data",
    xytext=(75, 75),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="black"),
    va="center",
    ha="center",
    fontsize=15,
)
sns.despine()
for n, ax in enumerate(axs):
    ax.text(
        -0.075,
        -0.15,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
# Save figure
fig.savefig(
    path_figures + "Figure_3.6.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -----------------------------------------------------------------------------
# Figures 3.7
# Number of interpolations required to standardize the data to1-hour intervals
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load data
df1 = pd.read_csv(
    "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/merged/raw.csv",
    index_col=False,
)

df2 = pd.read_csv(
    "/Volumes/data/nais_iceberg_drift_model/input/iceberg_database/merged/interpolated.csv",
    index_col=False,
)

# Get max number of interpolations
df3 = df2.groupby("beacon_id", as_index=False)["interpolated"].max()

# Get size of groups
df4 = df2.groupby("beacon_id", as_index=False).size()

# Plot interpolation alongside total number of observations
fig, ax = plt.subplots(figsize=(8, 12))
ax.grid(ls="dotted")
sns.barplot(y="beacon_id", x="size", data=df4, color="lightgrey", edgecolor="black")
sns.barplot(y="beacon_id", x="interpolated", data=df3, color="grey", edgecolor="black")
sns.despine()
ax.set(ylabel="Beacon ID", xlabel="Count")
top_bar = mpatches.Patch(color="lightgrey", label="observations")
bottom_bar = mpatches.Patch(color="grey", label="interpolations")
ax.legend(loc=4, handles=[top_bar, bottom_bar])
# Save figure
fig.savefig(
    path_figures + "Figure_3.7.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)

# ----------------------------------------------------------------------------
# Figure 3.8
# Example of  irregular observations interpolated to 1-hour interval
# Last confirmed working 2022-01-13
# ----------------------------------------------------------------------------

# Load data
df1 = pd.read_csv(path_data + "input/iceberg_database/merged/raw.csv", index_col=False)
df2 = pd.read_csv(
    path_data + "input/iceberg_database/merged/interpolated.csv", index_col=False
)

# Plot
plt.figure(figsize=(8, 4))
ax = plt.axes(projection=ccrs.Orthographic(-53.6, 50))
ax.add_feature(coast)
ax.set_extent([-53.75, -53.6, 50.09, 50.15])  # Drift loops
lon_formatter = LongitudeFormatter(number_format=".2f")
lat_formatter = LatitudeFormatter(number_format=".2f")
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.3,
    linestyle="dotted",
    xformatter=lon_formatter,
    yformatter=lat_formatter,
)
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False
ax.set_adjustable("datalim")

ax.scatter(
    x="longitude",
    y="latitude",
    color="blue",
    data=df1,
    label="Observed",
    s=200,
    marker="o",
    lw=1,
    facecolor="none",
    transform=ccrs.PlateCarree(),
)
ax.scatter(
    x="longitude",
    y="latitude",
    color="red",
    data=df2,
    label="Interpolated",
    s=200,
    marker="x",
    lw=2,
    facecolor="red",
    transform=ccrs.PlateCarree(),
)
ax.legend()
# Save figure
plt.savefig(
    path_figures + "Figure_3.8.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -----------------------------------------------------------------------------
# Figure 3.10
# Approximations of iceberg keel based on waterline length
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Calculates and illustrates the relationship between iceberg waterline length
# and iceberg keel depth based on equations used by CIS Fortran and NRC Python code
# (i.e., based on Barker et al., 2004)


def calculate_keel(length):
    draft_to_length = 1.0127 - 0.0020 * length
    if draft_to_length < 0.2:
        draft_to_length = 0.2
    return draft_to_length * length


# Sample calcualtion
calculate_keel(1000)

# Create an empty dataframe
keel = pd.DataFrame()

# Create sizes
keel["x"] = range(0, 1000 + 10, 10)

# Fortran
keel["y1"] = [i for i in map(calculate_keel, keel["x"])]

# Barker
keel["y2"] = keel["x"] * 0.7

# print(list(map(calculate_keel, x)))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.set(xlabel="Keel Depth (m)", ylabel="Iceberg Length (m)")
sns.lineplot(x="y1", y="x", data=keel, ci=None, sort=False, label="NAIS", color="red")
sns.lineplot(
    x="y2",
    y="x",
    data=keel,
    ci=None,
    sort=False,
    label="Barker et al., 2004",
    color="blue",
)
# ax.set_aspect("equal")
sns.despine()
ax.legend()

# Save figure
plt.savefig(
    path_figures + "Figure_3.10.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)

# -----------------------------------------------------------------------------
# Figure 3.12
# Scatterplot of cumluative distances vs. iceberg length
# Last confirmed working 2022-01-18
# -----------------------------------------------------------------------------

# Load data
df = pd.read_csv(path_data + "output/validation/merged/error_2009.csv", index_col=False)

# Add ocean current model column
df["current"] = df["branch"].str.split("_").str[1]
df1 = df[df["current"] == "cecom"]

# Load dimensions
df2 = pd.read_csv(path_data + "input/subset.csv", index_col=False)

# Change column names to lower case
df2 = df2.rename(columns={"beacon_id": "id"})

# Select required columns
df2 = df2[["id", "length", "keel"]]


fig, ax = plt.subplots(figsize=(8,5))
sns.regplot(x='keel', y='length', data=df2, ci=None,
            scatter_kws={"color": 'grey', 'edgecolor':'black'}, 
            line_kws={'color': 'k'})
ax.set(xlabel='Keel depth (m)', ylabel='Length (m)', )
ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
ax.yaxis.set_major_locator(mticker.MultipleLocator(250))
sns.despine()
# Save figure
fig.savefig(
    path_figures + "Figure_3.12.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)

# -----------------------------------------------------------------------------
# Figure 4.3
# Examples of variability in model performance
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load data
df = load_data(2017)

df["segment"] = df["branch"].str.split("_").str[0]
df["ensemble"] = df["id"] + "_" + df["segment"]

# Example 1
df1 = df[df["ensemble"].isin(["2019_1124-2670293_0"])]
df1_obs = df1[df1["branch"].isin(["0_cecom"])]
# Example 2
df2 = df[df["ensemble"].isin(["2017_300234060272000_2"])]
df2_obs = df2[df2["branch"].isin(["0_cecom"])]

label1 = ["Observed", "CECOM", "GLORYS", "RIOPS"]

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
axs[0].grid(ls="dotted")
axs[0].set_adjustable("datalim")
axs[0].set_aspect("equal")
axs[0].set(xlabel="East (km)", ylabel="North (km)")
axs[0].set_title(
    df1.iloc[0, df.columns.get_loc("id")],
    # str(df2["datetime"].min()) + " - " + str(df2["datetime"].max()),
    loc="left",
    fontweight="bold",
)
sns.lineplot(
    x="x_obs",
    y="y_obs",
    color="black",
    data=df1,
    markers=True,
    markevery=3,
    dashes=False,
    ci=None,
    sort=False,
    ax=axs[0],
)
sns.lineplot(
    x="x_model",
    y="y_model",
    data=df1,
    hue="branch",
    style="branch",
    markers=True,
    markevery=3,
    dashes=False,
    ci=None,
    sort=False,
    ax=axs[0],
)
axs[0].annotate(
    text="Start",
    xy=(2, -1),
    xycoords="data",
    fontsize=18,
    weight="bold",
    textcoords="offset points",
)
axs[0].legend(labels=label1)

axs[1].grid(ls="dotted")
axs[1].set_adjustable("datalim")
axs[1].set_aspect("equal")
axs[1].set(xlabel="East (km)", ylabel="North (km)")
axs[1].set_title(
    df2.iloc[0, df.columns.get_loc("id")],
    # str(df2["datetime"].min()) + " - " + str(df2["datetime"].max()),
    loc="left",
    fontweight="bold",
)
sns.lineplot(
    x="x_obs",
    y="y_obs",
    color="black",
    data=df2,
    markers=True,
    markevery=3,
    dashes=False,
    ci=None,
    sort=False,
    ax=axs[1],
)
sns.lineplot(
    x="x_model",
    y="y_model",
    data=df2,
    hue="branch",
    style="branch",
    markers=True,
    markevery=3,
    dashes=False,
    ci=None,
    sort=False,
    ax=axs[1],
)
axs[1].annotate(
    text="Start",
    xy=(-2, -3),
    xycoords="data",
    fontsize=18,
    weight="bold",
    textcoords="offset points",
)
axs[1].legend(labels=label1)

axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.1,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
# Save figure
fig.savefig(
    path_figures + "Figure_4.3.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)

# -------------------------------------------------------------------------
# Figure 4.4 & Figure 4.7
# Plots of distance error for all hindcasts + mean distance error
# Last confirmed working 2022-01-13
# -------------------------------------------------------------------------

# Interval between x-axis ticks
tick_spacing = 12

# CECOM/GLORYS

# Load data
df = load_data(2009)

# Subset data
df1 = df[df["current"] == "cecom"]
df2 = df[df["current"] == "glorys"]

# Plot
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6), constrained_layout=True)
axs[0].set(xlabel="Hindcast Duration (hours)", ylabel="Distance error (km)")
axs[0].set(box_aspect=1)
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
for label, group in df1.groupby(["ensemble"]):
    sns.lineplot(
        x="dur_obs",
        y="dist_error",
        data=group,
        color="grey",
        alpha=0.5,
        lw=1,
        ci=None,
        ax=axs[0],
    )
sns.lineplot(
    x="dur_obs", y="dist_error", color="red", lw=3, ci=None, data=df1, ax=axs[0]
)
axs[1].set(xlabel="Hindcast Duration (hours)", ylabel="Distance error (km)")
axs[1].set(box_aspect=1)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
for label, group in df2.groupby(["ensemble"]):
    sns.lineplot(
        x="dur_obs",
        y="dist_error",
        data=group,
        color="grey",
        alpha=0.5,
        lw=1,
        ci=None,
        ax=axs[1],
    )
sns.lineplot(
    x="dur_obs", y="dist_error", color="lime", lw=3, ci=None, data=df2, ax=axs[1]
)
axs[1].yaxis.set_tick_params(labelleft=True)
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.025,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="left",
        )
    if n == 1:
        ax.text(
            0.025,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="left",
        )
# Save figure
fig.savefig(
    path_figures + "Figure_4.4.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)

# CECOM/GLORYS/RIOPS

# Load data
df = load_data(2017)

# Subset data
df1 = df[df["current"] == "cecom"]
df2 = df[df["current"] == "glorys"]
df3 = df[df["current"] == "riops"]

# Plot
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18, 6), constrained_layout=True)
axs[0].set(xlabel="Hindcast Duration (hours)", ylabel="Distance error (km)")
axs[0].set(box_aspect=1)
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
for label, group in df1.groupby(["ensemble"]):
    sns.lineplot(
        x="dur_obs",
        y="dist_error",
        data=group,
        color="grey",
        alpha=0.5,
        lw=1,
        ci=None,
        ax=axs[0],
    )
sns.lineplot(
    x="dur_obs", y="dist_error", color="red", lw=3, ci=None, data=df1, ax=axs[0]
)
axs[1].set(xlabel="Hindcast Duration (hours)", ylabel="Distance error (km)")
axs[1].set(box_aspect=1)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
for label, group in df2.groupby(["ensemble"]):
    sns.lineplot(
        x="dur_obs",
        y="dist_error",
        data=group,
        color="grey",
        alpha=0.5,
        lw=1,
        ci=None,
        ax=axs[1],
    )
sns.lineplot(
    x="dur_obs", y="dist_error", color="lime", lw=3, ci=None, data=df2, ax=axs[1]
)
axs[1].yaxis.set_tick_params(labelleft=True)
axs[2].set(xlabel="Hindcast Duration (hours)", ylabel="Distance error (km)")
axs[2].set(box_aspect=1)
axs[2].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
for label, group in df3.groupby(["ensemble"]):
    sns.lineplot(
        x="dur_obs",
        y="dist_error",
        data=group,
        color="grey",
        alpha=0.5,
        lw=1,
        ci=None,
        ax=axs[2],
    )
sns.lineplot(
    x="dur_obs", y="dist_error", color="blue", lw=3, ci=None, data=df3, ax=axs[2]
)
axs[2].yaxis.set_tick_params(labelleft=True)
sns.despine()
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.025,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="left",
        )
    if n == 1:
        ax.text(
            0.025,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="left",
        )
    if n == 2:
        ax.text(
            0.025,
            0.925,
            "RIOPS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="left",
        )
# Save figure
fig.savefig(
    path_figures + "Figure_4.7.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)

# -------------------------------------------------------------------------
# Figure 4.5
# Histograms of final distance error and cumulative distance error
# Last confirmed working 2022-01-16
# -------------------------------------------------------------------------

# Interval between x-axis ticks
tick_spacing = 25

# CECOM/GLORYS

# Load data
df = load_data(2009)

# Subset last value
#df = df.sort_values("datetime").groupby("ensemble").tail(1)

# Subset data
df = df[df['dur_obs'] == 96]
df1 = df[df["current"] == "cecom"]
df2 = df[df["current"] == "glorys"]

# Plot final distance error
fig, axs = plt.subplots(
    2, 1, sharex=True, sharey=True, figsize=(6, 6), constrained_layout=True
)
axs[0].set(ylabel="Frequency")
axs[0].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_error",
    data=df1,
    binwidth=5,
    color="grey",
    edgecolor="black",
    ax=axs[0],
    label="CECOM",
)
axs[0].axvline(x=df1.dist_error.mean(), color="red", ls="--", lw=2.5)
axs[1].set(xlabel="Final Distance Error (km)", ylabel="Frequency")
axs[1].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_error",
    data=df2,
    binwidth=5,
    color="grey",
    edgecolor="black",
    ax=axs[1],
    label="GLORYS",
)
axs[1].axvline(x=df2.dist_error.mean(), color="lime", ls="--", lw=2.5)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.2,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.925,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 1:
        ax.text(
            0.925,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
# Save figure
plt.savefig(
    path_figures + "Figure_4.5ab.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)


# Plot cumulative distance error
fig, axs = plt.subplots(
    2, 1, sharex=True, sharey=True, figsize=(6, 6), constrained_layout=True
)
axs[0].set(ylabel="Frequency")
axs[0].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_csum_error",
    data=df1,
    binwidth=5,
    binrange=(-50, 175),
    color="grey",
    edgecolor="black",
    ax=axs[0],
    label="CECOM",
)
axs[0].axvline(x=df1.dist_csum_error.mean(), color="red", ls="--", lw=2.5)
axs[1].set(xlabel="Cumulative Distance Error (km)", ylabel="Frequency")
axs[1].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_csum_error",
    data=df2,
    binwidth=5,
    binrange=(-50, 175),
    color="grey",
    edgecolor="black",
    ax=axs[1],
    label="GLORYS",
)
axs[1].axvline(x=df2.dist_csum_error.mean(), color="lime", ls="--", lw=2.5)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.2,
        "(" + string.ascii_lowercase[n + 2] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.925,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 1:
        ax.text(
            0.925,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
# Save figure
plt.savefig(
    path_figures + "Figure_4.5cd.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)


# -----------------------------------------------------------------------------
# Figure 4.6 & 4.9
# Plot RMSE & 24-hour RMSE
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# CECOM/GLORYS

# Load data
df = load_data(2009)

# Calculate statistics
stats = calculate_statistics(df)

# Subset data
stats_24 = stats[stats["dur_obs"] <= 24]

# Plot
labels = ["CECOM", "GLORYS"]
fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
axs[0].set(xlabel="Hindcast Duration (hours)", ylabel="RMSE (km)")
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(12))
axs[0].set(box_aspect=1)
sns.lineplot(
    x="dur_obs",
    y="rmse",
    data=stats,
    hue="current",
    style="current",
    ci=None,
    ax=axs[0],
)
axs[0].add_patch(Rectangle((0, 0), 24, 21, fc="none", ec="k", lw=2, ls="dotted"))

axs[0].legend(loc=4, labels=labels)
axs[1].set(xlabel="Hindcast Duration (hours)", ylabel="RMSE (km)")
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(4))
axs[1].set(box_aspect=1)
sns.lineplot(
    x="dur_obs",
    y="rmse",
    data=stats_24,
    hue="current",
    style="current",
    ci=None,
    ax=axs[1],
)
sns.despine()
axs[1].legend(loc=4, labels=labels)
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
fig.savefig(
    path_figures + "Figure_4.6.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# CECOM/GLORYS/RIOPS

# Load data
df = load_data(2017)

# Calculate statistics
stats = calculate_statistics(df)

# Subset data
stats_24 = stats[stats["dur_obs"] <= 24]

# Plot
labels = ["CECOM", "GLORYS", "RIOPS"]
fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
axs[0].set(xlabel="Hindcast Duration (hours)", ylabel="RMSE (km)")
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(12))
axs[0].set(box_aspect=1)
sns.lineplot(
    x="dur_obs",
    y="rmse",
    data=stats,
    hue="current",
    style="current",
    ci=None,
    ax=axs[0],
)
axs[0].add_patch(Rectangle((0, 0), 24, 21, fc="none", ec="k", lw=2, ls="dotted"))
sns.despine()
axs[0].legend(loc=4, labels=labels)
axs[1].set(xlabel="Hindcast Duration (hours)", ylabel="RMSE (km)")
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(4))
axs[1].set(box_aspect=1)
sns.lineplot(
    x="dur_obs",
    y="rmse",
    data=stats_24,
    hue="current",
    style="current",
    ci=None,
    ax=axs[1],
)
sns.despine()
axs[1].legend(loc=4, labels=labels)
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
fig.savefig(
    path_figures + "Figure_4.9.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -----------------------------------------------------------------------------
# Figure 4.8
# Histograms of final and cumulative distance error
# Last confirmed working 2022-01-16
# -----------------------------------------------------------------------------

# Set tick spacing
tick_spacing = 25

# CECOM/GLORYS/RIOPS

# Load data
df = load_data(2017)

# Subset data
#df = df.sort_values("datetime").groupby("ensemble").tail(1)
df = df[df['dur_obs'] == 96]

df1 = df[df["current"] == "cecom"]
df2 = df[df["current"] == "glorys"]
df3 = df[df["current"] == "riops"]

# Plot final distance error
fig, axs = plt.subplots(
    3, sharex=True, sharey=True, figsize=(6, 9), constrained_layout=True
)
axs[0].set(ylabel="Frequency")
axs[0].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_error",
    data=df1,
    binwidth=5,
    binrange=(0, 150),
    color="grey",
    edgecolor="black",
    ax=axs[0],
    label="CECOM",
)
axs[0].axvline(x=df1.dist_error.mean(), color="red", ls="--", lw=2.5)
axs[1].set(ylabel="Frequency")
axs[1].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_error",
    data=df2,
    binwidth=5,
    binrange=(0, 150),
    color="grey",
    edgecolor="black",
    ax=axs[1],
    label="GLORYS",
)
axs[1].axvline(x=df2.dist_error.mean(), color="lime", ls="--", lw=2.5)
axs[2].set(xlabel="Final Distance Error (km)", ylabel="Frequency")
axs[2].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_error",
    data=df3,
    binwidth=5,
    binrange=(0, 150),
    color="grey",
    edgecolor="black",
    ax=axs[2],
    label="RIOPS",
)
axs[2].axvline(x=df2.dist_error.mean(), color="blue", ls="--", lw=2.5)
axs[2].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.2,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.925,  # Horizontal
            0.925,  # Vertical
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 1:
        ax.text(
            0.925,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 2:
        ax.text(
            0.925,
            0.925,
            "RIOPS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
plt.savefig(
    path_figures + "Figure_4.8abc.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)


# Plot cumulative distance error
fig, axs = plt.subplots(
    3, sharex=True, sharey=True, figsize=(6, 9), constrained_layout=True
)
axs[0].set(ylabel="Frequency")
axs[0].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_csum_error",
    data=df1,
    binwidth=5,
    binrange=(-50, 125),
    color="grey",
    edgecolor="black",
    ax=axs[0],
    label="CECOM",
)
axs[0].axvline(x=df1.dist_csum_error.mean(), color="red", ls="--", lw=2.5)
axs[1].set(ylabel="Frequency")
axs[1].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_csum_error",
    data=df2,
    binwidth=5,
    binrange=(-50, 125),
    color="grey",
    edgecolor="black",
    ax=axs[1],
    label="GLORYS",
)
axs[1].axvline(x=df2.dist_csum_error.mean(), color="lime", ls="--", lw=2.5)
axs[2].set(xlabel="Cumulative Distance Error (km)", ylabel="Frequency")
axs[2].grid(axis="x", ls="dotted")
sns.histplot(
    x="dist_csum_error",
    data=df3,
    binwidth=5,
    binrange=(-50, 125),
    color="grey",
    edgecolor="black",
    ax=axs[2],
    label="RIOPS",
)
axs[2].axvline(x=df2.dist_csum_error.mean(), color="blue", ls="--", lw=2.5)
axs[2].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.2,
        "(" + string.ascii_lowercase[n + 3] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.925,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 1:
        ax.text(
            0.925,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 2:
        ax.text(
            0.925,
            0.925,
            "RIOPS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
plt.savefig(
    path_figures + "Figure_4.8def.eps",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)

# -----------------------------------------------------------------------------
# Figure 4.10
# Boxplots of final and cumulative distance error vs season
# Last confirmed working 2022-01-17
# -----------------------------------------------------------------------------

# Function to associate month numbers with season names
def find_season(month, hemisphere):
    if hemisphere == "Southern":
        season_month_south = {
            12: "Summer",
            1: "Summer",
            2: "Summer",
            3: "Fall",
            4: "Fall",
            5: "Fall",
            6: "Winter",
            7: "Winter",
            8: "Winter",
            9: "Spring",
            10: "Spring",
            11: "Spring",
        }
        return season_month_south.get(month)

    elif hemisphere == "Northern":
        season_month_north = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
        return season_month_north.get(month)
    else:
        print("Invalid selection. Please select a hemisphere and try again")


# Load data
df1 = load_data(2009)
df2 = load_data(2017)

# Subset head and tail values
df1 = df1.sort_values(["datetime", "current"]).groupby("ensemble").tail(1)
df2 = df2.sort_values(["datetime", "current"]).groupby("ensemble").tail(1)

# Convert to datetime
df1["datetime"] = pd.to_datetime(
    df1["datetime"].astype(str), format="%Y-%m-%d %H:%M:%S"
)
df1["month_int"] = df1["datetime"].dt.month
df1["month_str"] = (
    pd.to_datetime(df1["month_int"], format="%m").dt.month_name().str.slice(stop=3)
)
df1 = df1.sort_values(by="month_int")
df2["datetime"] = pd.to_datetime(
    df2["datetime"].astype(str), format="%Y-%m-%d %H:%M:%S"
)
df2["month_int"] = df2["datetime"].dt.month
df2["month_str"] = (
    pd.to_datetime(df2["month_int"], format="%m").dt.month_name().str.slice(stop=3)
)
df2 = df2.sort_values(by="month_int")


# Convert months to seasons
season_list1 = []
hemisphere = "Northern"
for month in df1["month_int"]:
    season1 = find_season(month, hemisphere)
    season_list1.append(season1)
season_list2 = []
for month in df2["month_int"]:
    season2 = find_season(month, hemisphere)
    season_list2.append(season2)

# Add season column
df1["season"] = season_list1
df2["season"] = season_list2

# Change to upper case
df1["current"] = df1["current"].str.upper()
df2["current"] = df2["current"].str.upper()


# Plot
fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
sns.boxplot(x="season", y="dist_error", hue="current", data=df1, ax=axs[0, 0])
axs[0, 0].set(xlabel=None, ylabel="Final Distance Error (km)", ylim=(-3, 165))
axs[0, 0].set(box_aspect=1)
axs[0, 0].get_legend().remove()
sns.boxplot(x="season", y="dist_csum_error", hue="current", data=df1, ax=axs[0, 1])
axs[0, 1].set(xlabel=None, ylabel="Cumulative Distance Error (km)", ylim=(-70, 170))
axs[0, 1].legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
axs[0, 1].set(box_aspect=1)
sns.boxplot(x="season", y="dist_error", hue="current", data=df2, ax=axs[1, 0])
axs[1, 0].set(xlabel=None, ylabel="Final Distance Error (km)", ylim=(-3, 165))
axs[1, 0].set(box_aspect=1)
axs[1, 0].get_legend().remove()
sns.boxplot(x="season", y="dist_csum_error", hue="current", data=df2, ax=axs[1, 1])
axs[1, 1].set(xlabel=None, ylabel="Cumulative Distance Error (km)", ylim=(-70, 170))
axs[1, 1].legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
axs[1, 1].set(box_aspect=1)
sns.despine()
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
plt.savefig(path_figures + "Figure_4.10.eps", dpi=dpi, transparent=False)

# -----------------------------------------------------------------------------
# Figure 4.11
# Scatterplot - Cumluative distance vs. Length
# Last confirmed working 2022-01-17
# -----------------------------------------------------------------------------

# Load data
df = load_data(2009)

# Load dimensions
df2 = pd.read_csv(path_data + "input/subset.csv", index_col=False)

# Change column names to lower case
df2.columns = map(str.lower, df2.columns)
df2 = df2.rename(columns={"beacon_id": "id"})

# Select only 2 columns
df2 = df2[["id", "length", "keel"]]

# Merge dataframes
df3 = pd.merge(df1, df2, on="id")

# Subset head and tail values
df4 = df3.sort_values("datetime").groupby("ensemble").tail(1)

# Select only 96-hour observations to avoid bias
df4 = df4[df4["dur_obs"] == 96]

# Plot csum
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(
    x="length",
    y="dist_csum_obs",
    data=df4,
    scatter_kws={"color": "grey", "edgecolor": "black"},
    line_kws={"color": "k"},
)
ax.set(
    xlabel="Length (m)",
    ylabel="Cumulative Distance (km)",
    xlim=(0, 1600),
    ylim=(-3, 225),
)
ax.xaxis.set_major_locator(mticker.MultipleLocator(250))
sns.despine()

# Save figure
fig.savefig(
    path_figures + "Figure_4.11.png", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -----------------------------------------------------------------------------
# Figure 4.12
# Scatterplot of RMSE as a function of iceberg waterline length
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load data
df1 = load_data(2009)
df2 = load_data(2017)

# Change current models to uppercase
df1["current"] = df1["current"].str.upper()
df2["current"] = df2["current"].str.upper()

# Calculate RMSE for each hindcast
stats_track_2009 = (
    df1.groupby(["ensemble", "current"])["dist_error"]
    .apply(rmse)
    .reset_index(name="rmse")
)
stats_track_2017 = (
    df2.groupby(["ensemble", "current"])["dist_error"]
    .apply(rmse)
    .reset_index(name="rmse")
)

# Load iceberg dimensions
df3 = pd.read_csv(path_data + "/input/subset.csv", index_col=False)

# Change column names
df3 = df3.rename(columns={"beacon_id": "id"})

# Select required columns
df4 = df3[["id", "length", "keel"]]

# Merge dimensions
df_2009 = pd.merge(df1, df4, on="id")
df_2017 = pd.merge(df2, df4, on="id")

# Add id columns
df4["id"] = df4["id"].str.split("_").str[1]

# Extract id from ensemble column
stats_track_2009["id"] = stats_track_2009["ensemble"].str.split("_").str[1]
stats_track_2017["id"] = stats_track_2017["ensemble"].str.split("_").str[1]

# Merge stats based on id
temp_2009 = pd.merge(df4, stats_track_2009, on="id")
temp_2017 = pd.merge(df4, stats_track_2017, on="id")

# Plot looping regplots 2009 & 2017
_, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)
for d, m, ls in zip(temp_2009["current"].unique(), ["o", "X", "s"], ["-", "--", "-."]):
    sns.regplot(
        x="length",
        y="rmse",
        data=temp_2009.loc[temp_2009.current == d],
        marker=m,
        scatter_kws={"linewidths": 1, "edgecolor": "k"},
        line_kws={"ls": ls},
        ax=axs[0],
        label=d,
    )
axs[0].set(xlabel="Length (m)", ylabel="RMSE (km)", xlim=(0, 1600))
axs[0].set(box_aspect=1)
# axs[0].legend(loc=1)
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(250))
for d, m, ls in zip(temp_2017["current"].unique(), ["o", "X", "s"], ["-", "--", "-."]):
    sns.regplot(
        x="length",
        y="rmse",
        data=temp_2017.loc[temp_2017.current == d],
        marker=m,
        scatter_kws={"linewidths": 1, "edgecolor": "k"},
        line_kws={"ls": ls},
        ax=axs[1],
        label=d,
    )
axs[1].set(xlabel="Length (m)", ylabel="RMSE (km)", xlim=(0, 450))
axs[1].set(box_aspect=1)
axs[1].yaxis.set_tick_params(labelleft=True)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(50))
sns.despine()
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.15,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=25,
        weight="bold",
    )

plt.savefig(
    path_figures + "Figure_4.12.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)


# -----------------------------------------------------------------------------
# Figure 4.13
# Scatterplot of RMSE as a function of starting latitude
# Last confirmed working 2022-01-16
# -----------------------------------------------------------------------------

# Load data
df1 = load_data(2009)
df2 = load_data(2017)

# Change current models to uppercase
df1["current"] = df1["current"].str.upper()
df2["current"] = df2["current"].str.upper()

# Calculate RMSE for each hindcast
stats_track_2009 = (
    df1.groupby(["ensemble", "current"])["dist_error"]
    .apply(rmse)
    .reset_index(name="rmse")
)
stats_track_2017 = (
    df2.groupby(["ensemble", "current"])["dist_error"]
    .apply(rmse)
    .reset_index(name="rmse")
)

# Subset head and tail values
start_2009 = df1.sort_values(["datetime", "branch"]).groupby("ensemble").head(1)
start_2017 = df2.sort_values(["datetime", "branch"]).groupby("ensemble").head(1)

end_2009 = df1.sort_values(["datetime", "branch"]).groupby("ensemble").tail(1)
end_2017 = df2.sort_values(["datetime", "branch"]).groupby("ensemble").tail(1)

# Create two new dataframes
temp1 = pd.DataFrame()
temp1["ensemble"] = start_2009["ensemble"]
temp1["latitude"] = start_2009["lat_obs"]
temp1["current"] = start_2009["current"]

temp2 = pd.DataFrame()
temp2["ensemble"] = end_2009["ensemble"]
temp2["dist_error"] = end_2009["dist_error"]

temp3 = pd.DataFrame()
temp3["ensemble"] = start_2017["ensemble"]
temp3["latitude"] = start_2017["lat_obs"]
temp3["current"] = start_2017["current"]

temp4 = pd.DataFrame()
temp4["ensemble"] = end_2017["ensemble"]
temp4["dist_error"] = end_2017["dist_error"]

# Merge dataframes
result1 = pd.merge(temp1, temp2, on="ensemble")
result2 = pd.merge(temp3, temp4, on="ensemble")

# Merge RMSE and starting latitude dataframes
result3 = pd.merge(result1, stats_track_2009, on=["ensemble", "current"])
result4 = pd.merge(result2, stats_track_2017, on=["ensemble", "current"])

# Plot looping regplots 2009 & 2017
_, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)
for d, m, ls in zip(result3["current"].unique(), ["o", "X", "s"], ["-", "--", "-."]):
    sns.regplot(
        x="latitude",
        y="rmse",
        data=result3.loc[result3.current == d],
        marker=m,
        scatter_kws={"linewidths": 1, "edgecolor": "k"},
        line_kws={"ls": ls},
        ax=axs[0],
        label=d,
    )
axs[0].set(xlabel="Starting Latitude (°)", ylabel="RMSE (km)", xlim=(44, 82))
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
axs[0].set(box_aspect=1)
# axs[0].legend(loc=1)

for d, m, ls in zip(result4["current"].unique(), ["o", "X", "s"], ["-", "--", "-."]):
    sns.regplot(
        x="latitude",
        y="rmse",
        data=result4.loc[result4.current == d],
        marker=m,
        scatter_kws={"linewidths": 1, "edgecolor": "k"},
        line_kws={"ls": ls},
        ax=axs[1],
        label=d,
    )
axs[1].set(xlabel="Starting Latitude (°)", ylabel="RMSE (km)", xlim=(44, 82))
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
axs[1].set(box_aspect=1)
axs[1].yaxis.set_tick_params(labelleft=True)
# axs[1].legend(loc=1)
sns.despine()
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.15,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=25,
        weight="bold",
    )

plt.savefig(
    path_figures + "Figure_4.13.png", dpi=dpi, transparent=False, bbox_inches="tight"
)


# ----------------------------------------------------------------------------
# Figure 4.14
# 4 panel plot of CECOM/GLORYS 24-hour RMSE
# Last confirmed working 2022-01-16
# ----------------------------------------------------------------------------

# Load data
df = load_data(2009)

# Read grid shapefile
grid = gpd.read_file(
    "/Volumes/GoogleDrive/My Drive/University of Ottawa/Thesis/NAIS Iceberg Drift Model/input/shp/100_km_grid.shp"
)

# Create grid coordinates (cheap centroid)
grid["coords"] = grid["geometry"].apply(lambda x: x.representative_point().coords[:])
grid["coords"] = [coords[0] for coords in grid["coords"]]

# Subset by duration
df = df[df["dur_obs"] == 24]

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(df.lon_model, df.lat_model)]
gdf = GeoDataFrame(df, crs="epsg:4326", geometry=geometry)

# Reproject data to EPSG 3347
gdf = gdf.to_crs(epsg=3347)

# Spatial join grid with points
joined = gpd.sjoin(gdf, grid, how="left", op="within")

# Calculate statistics
rmse_long = joined.groupby(["index_right", "current"])["dist_error"].apply(rmse)
count_long = joined.groupby(["index_right", "current"])["ensemble"].agg(["nunique"])

# Get max values for colorbar normalization
norm_max = np.nanmax(rmse_long)

# Unstack
rmse_wide = rmse_long.unstack()
count_wide = count_long.unstack()

# Merge dataframes
merged = pd.merge(grid, rmse_wide, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, count_wide, left_index=True, right_index=True, how="outer")


# Normalize colourbar
norm = mpl.colors.Normalize(vmin=0, vmax=50)
cmap = cm.get_cmap("turbo", 20)

# Set map projection
proj = ccrs.epsg(3347)

# Set extents of map >70°N
# extents = [-74, -73, 70.5, 81]
extents = [-75, -65, 64.5, 81]

# Plot figures (N = 11,6) (S = 13.5)
fig, axs = plt.subplots(
    2, 2, figsize=(14, 12), constrained_layout=True, subplot_kw={"projection": proj}
)
axs[0, 0].add_feature(coast)
axs[0, 0].set_extent(extents)
axs[0, 0].set(box_aspect=1)
p1 = merged.plot(
    column="cecom",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[0, 0],
)
gl1 = axs[0, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl1.top_labels = False
gl1.right_labels = False
gl1.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "cecom")]):
        p1.annotate(
            text=("%i" % row[("nunique", "cecom")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )
axs[0, 1].add_feature(coast)
axs[0, 1].set_extent(extents)
axs[0, 1].set(box_aspect=1)
p2 = merged.plot(
    column="glorys",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[0, 1],
)
gl = axs[0, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "glorys")]):
        p2.annotate(
            text=("%i" % row[("nunique", "glorys")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

# Set extents of map <60°N
extents = [-64.5, -48.5, 49, 53]

axs[1, 0].add_feature(coast)
axs[1, 0].set_extent(extents)
axs[1, 0].set(box_aspect=1)
p3 = merged.plot(
    column="cecom",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[1, 0],
)
gl = axs[1, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "cecom")]):
        p3.annotate(
            text=("%i" % row[("nunique", "cecom")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

axs[1, 1].add_feature(coast)
axs[1, 1].set_extent(extents)
axs[1, 1].set(box_aspect=1)
p4 = merged.plot(
    column="glorys",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[1, 1],
)
gl4 = axs[1, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl4.top_labels = False
gl4.right_labels = False
gl4.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "glorys")]):
        p4.annotate(
            text=("%i" % row[("nunique", "glorys")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )
# Flatten axes
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.1,
        -0.075,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=30,
        weight="bold",
    )

    if n == 0 or n == 2:
        ax.text(
            0.95,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 1 or n == 3:
        ax.text(
            0.95,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )

# Add colour bar
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.8)

# Save figure
fig.savefig(path_figures + "Figure_4.14.png", dpi=dpi, transparent=False)


# ----------------------------------------------------------------------------
# Figure 4.15
# 6-panel plot of CECOM/GLORYS/RIOPS 24-hour RMSE
# Last confirmed working 2022-01-16
# ----------------------------------------------------------------------------

# Load data
df = load_data(2017)

# Load grid shapefile
grid = gpd.read_file(path_data + "/input/shp/100_km_grid.shp")

# Create grid coordinates (cheap centroid)
grid["coords"] = grid["geometry"].apply(lambda x: x.representative_point().coords[:])
grid["coords"] = [coords[0] for coords in grid["coords"]]

# Subset by duration
df = df[df["dur_obs"] == 24]

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(df.lon_model, df.lat_model)]
gdf = GeoDataFrame(df, crs="epsg:4326", geometry=geometry)

# Reproject data to EPSG 3347
gdf = gdf.to_crs(epsg=3347)

# Spatial join grid with points
joined = gpd.sjoin(gdf, grid, how="left", op="within")

# Calculate statistics
rmse_long = joined.groupby(["index_right", "current"])["dist_error"].apply(rmse)
count_long = joined.groupby(["index_right", "current"])["ensemble"].agg(["nunique"])

# Get max values for colorbar normalization
norm_max = np.nanmax(rmse_long)

# Unstack
rmse_wide = rmse_long.unstack()
count_wide = count_long.unstack()

# Merge dataframes
merged = pd.merge(grid, rmse_wide, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, count_wide, left_index=True, right_index=True, how="outer")

# Set extents of map
# extents = [-90, -65, 70, 81] # Old

# Normalize colourbar
norm = mpl.colors.Normalize(vmin=0, vmax=50)
cmap = cm.get_cmap("turbo", 20)

# Set map projection
proj = ccrs.epsg(3347)

# Set extents of map >70°N
# extents = [-74, -73, 70.5, 81]
extents = [-75, -65, 66, 79]

# Plot figures (N = 1,6) (S = 19,6)
fig, axs = plt.subplots(
    2, 3, figsize=(21, 12), constrained_layout=True, subplot_kw={"projection": proj}
)
axs[0, 0].add_feature(coast)
axs[0, 0].set_extent(extents)
axs[0, 0].set(box_aspect=1)
p1 = merged.plot(
    column="cecom",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[0, 0],
)
gl1 = axs[0, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl1.top_labels = False
gl1.right_labels = False
gl1.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "cecom")]):
        p1.annotate(
            text=("%i" % row[("nunique", "cecom")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

axs[0, 1].add_feature(coast)
axs[0, 1].set_extent(extents)
axs[0, 1].set(box_aspect=1)
p2 = merged.plot(
    column="glorys",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[0, 1],
)
gl2 = axs[0, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl2.top_labels = False
gl2.right_labels = False
gl2.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "glorys")]):
        p2.annotate(
            text=("%i" % row[("nunique", "glorys")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

axs[0, 2].add_feature(coast)
axs[0, 2].set_extent(extents)
axs[0, 2].set(box_aspect=1)
p3 = merged.plot(
    column="riops",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[0, 2],
)
gl3 = axs[0, 2].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl3.top_labels = False
gl3.right_labels = False
gl3.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "riops")]):
        p3.annotate(
            text=("%i" % row[("nunique", "riops")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

# Set extents of map <60°N
extents = [-64.5, -48.5, 49, 53]

axs[1, 0].add_feature(coast)
axs[1, 0].set_extent(extents)
axs[1, 0].set(box_aspect=1)
p4 = merged.plot(
    column="cecom",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[1, 0],
)
gl4 = axs[1, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl4.top_labels = False
gl4.right_labels = False
gl4.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "cecom")]):
        p4.annotate(
            text=("%i" % row[("nunique", "cecom")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

axs[1, 1].add_feature(coast)
axs[1, 1].set_extent(extents)
axs[1, 1].set(box_aspect=1)
p5 = merged.plot(
    column="glorys",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[1, 1],
)
gl5 = axs[1, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl5.top_labels = False
gl5.right_labels = False
gl5.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "glorys")]):
        p5.annotate(
            text=("%i" % row[("nunique", "glorys")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )

axs[1, 2].add_feature(coast)
axs[1, 2].set_extent(extents)
axs[1, 2].set(box_aspect=1)
p6 = merged.plot(
    column="riops",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5,
    legend=False,
    ax=axs[1, 2],
)
gl6 = axs[1, 2].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl6.top_labels = False
gl6.right_labels = False
gl6.rotate_labels = False

for idx, row in merged.iterrows():
    if not pd.isnull(row[("nunique", "riops")]):
        p6.annotate(
            text=("%i" % row[("nunique", "riops")]),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            xy=row["coords"],
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="bold",
            size=12,
            linespacing=2,
        )
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.075,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=30,
        weight="bold",
    )
    if n == 0 or n == 3:
        ax.text(
            0.95,
            0.925,
            "CECOM",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 1 or n == 4:
        ax.text(
            0.95,
            0.925,
            "GLORYS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
    if n == 2 or n == 5:
        ax.text(
            0.95,
            0.925,
            "RIOPS",
            transform=ax.transAxes,
            size=20,
            weight="bold",
            ha="right",
        )
# Add colour bar
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.8)

# Save figure
fig.savefig(path_figures + "Figure_4.15.png", dpi=dpi, transparent=False)


# -----------------------------------------------------------------------------
# Figure 4.16
# Plot of distance error pairwise comparison delta for sensitivity analysis
# Note: 12 subplots that use a custom color palette (omits blue)
# Last confirmed working: 2022-01-17
# -----------------------------------------------------------------------------

# Set colour palette
colour = ["red", "lime", "magenta", "cyan", "yellow"]
sns.set_palette(colour)

path_stats = path_data + "output/sensitivity_analysis/statistics/distance_error/delta/"

# Find all files in folder
files = glob.glob(path_stats + "*.csv")

variable_list = [
    "length",
    "keel",
    "air_drag",
    "water_drag",
    "wind_direction",
    "wind_speed",
    "current_direction",
    "current_speed",
    "wind_wave_height",
    "wind_wave_direction",
    "swell_wave_height",
    "swell_wave_direction",
]

# Load data and set parameter column to type object
df1 = pd.read_csv(
    path_stats + variable_list[0] + ".csv", index_col=False, dtype={"parameter": object}
)
df2 = pd.read_csv(
    path_stats + variable_list[1] + ".csv", index_col=False, dtype={"parameter": object}
)
df3 = pd.read_csv(
    path_stats + variable_list[2] + ".csv", index_col=False, dtype={"parameter": object}
)
df4 = pd.read_csv(
    path_stats + variable_list[3] + ".csv", index_col=False, dtype={"parameter": object}
)
df5 = pd.read_csv(
    path_stats + variable_list[4] + ".csv", index_col=False, dtype={"parameter": object}
)
df6 = pd.read_csv(
    path_stats + variable_list[5] + ".csv", index_col=False, dtype={"parameter": object}
)
df7 = pd.read_csv(
    path_stats + variable_list[6] + ".csv", index_col=False, dtype={"parameter": object}
)
df8 = pd.read_csv(
    path_stats + variable_list[7] + ".csv", index_col=False, dtype={"parameter": object}
)
df9 = pd.read_csv(
    path_stats + variable_list[8] + ".csv", index_col=False, dtype={"parameter": object}
)
df10 = pd.read_csv(
    path_stats + variable_list[9] + ".csv", index_col=False, dtype={"parameter": object}
)
df11 = pd.read_csv(
    path_stats + variable_list[10] + ".csv",
    index_col=False,
    dtype={"parameter": object},
)
df12 = pd.read_csv(
    path_stats + variable_list[11] + ".csv",
    index_col=False,
    dtype={"parameter": object},
)

# ------------------------------------

replacement_mapping_dict = {"0.5": "50%", "0.75": "75%", "1.25": "125%", "1.5": "150%"}

df1["parameter"] = df1["parameter"].replace(replacement_mapping_dict)
df2["parameter"] = df2["parameter"].replace(replacement_mapping_dict)
df6["parameter"] = df6["parameter"].replace(replacement_mapping_dict)
df8["parameter"] = df8["parameter"].replace(replacement_mapping_dict)
df9["parameter"] = df9["parameter"].replace(replacement_mapping_dict)
df11["parameter"] = df11["parameter"].replace(replacement_mapping_dict)

replacement_mapping_dict = {"+60": "+60°", "+30": "+30°", "-30": "-30°", "-60": "-60°"}

df5["parameter"] = df5["parameter"].replace(replacement_mapping_dict)
df7["parameter"] = df7["parameter"].replace(replacement_mapping_dict)
df10["parameter"] = df10["parameter"].replace(replacement_mapping_dict)
df12["parameter"] = df12["parameter"].replace(replacement_mapping_dict)

# ------------------------------------

# Create figure and add axes object (nrows x ncols)
fig, axs = plt.subplots(4, 3, figsize=(13, 18), sharey=True, constrained_layout=True)
sns.lineplot(
    x="dur_obs", y="delta", data=df1, hue="parameter", style="parameter", ax=axs[0, 0]
).set_title("Iceberg length")
sns.lineplot(
    x="dur_obs", y="delta", data=df2, hue="parameter", style="parameter", ax=axs[0, 1]
).set_title("Iceberg keel depth")
sns.lineplot(
    x="dur_obs", y="delta", data=df3, hue="parameter", style="parameter", ax=axs[0, 2]
).set_title("Air drag coefficient")
sns.lineplot(
    x="dur_obs", y="delta", data=df4, hue="parameter", style="parameter", ax=axs[1, 0]
).set_title("Water drag coefficient")
sns.lineplot(
    x="dur_obs", y="delta", data=df5, hue="parameter", style="parameter", ax=axs[1, 1]
).set_title("Wind direction")
sns.lineplot(
    x="dur_obs", y="delta", data=df6, hue="parameter", style="parameter", ax=axs[1, 2]
).set_title("Wind speed")
sns.lineplot(
    x="dur_obs", y="delta", data=df7, hue="parameter", style="parameter", ax=axs[2, 0]
).set_title("Current direction")
sns.lineplot(
    x="dur_obs", y="delta", data=df8, hue="parameter", style="parameter", ax=axs[2, 1]
).set_title("Current Speed")
sns.lineplot(
    x="dur_obs", y="delta", data=df9, hue="parameter", style="parameter", ax=axs[2, 2]
).set_title("Wind wave height")
sns.lineplot(
    x="dur_obs", y="delta", data=df10, hue="parameter", style="parameter", ax=axs[3, 0]
).set_title("Wind wave direction")
sns.lineplot(
    x="dur_obs", y="delta", data=df11, hue="parameter", style="parameter", ax=axs[3, 1]
).set_title("Swell wave height")
sns.lineplot(
    x="dur_obs", y="delta", data=df12, hue="parameter", style="parameter", ax=axs[3, 2]
).set_title("Swell wave direction")
for ax in axs.ravel():  # Ravel axes to a flattened array
    ax.axhline(0, ls="-", color="black", alpha=0.75, lw=1, zorder=0)
    ax.set(box_aspect=1, xlabel=None, ylabel=None)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.legend(loc=2)
sns.despine()
fig.supxlabel("Hindcast Duration (hours)")
fig.supylabel("Distance Error (km)")
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
# Save figure
fig.savefig(path_figures + "Figure_4.16.png", dpi=dpi, transparent=False)


# -----------------------------------------------------------------------------
# Figures 4.17 - 4.26
# Sensitivity analysis iceberg trajectory plots
# Last confirmed working 2022-01-17
# -----------------------------------------------------------------------------


# List of variables to process
variable_list = [
    "length",
    "keel",
    "air_drag",
    "water_drag",
    "wind_direction",
    # "wind_speed",
    "current_direction",
    "current_speed",
    "wind_wave_direction",
    # "wind_wave_height",
    "swell_wave_height",
    "swell_wave_direction",
]

# List of figure names
figure_list = [
    "Figure_4.17",
    "Figure_4.18",
    "Figure_4.19",
    "Figure_4.20",
    "Figure_4.21",
    "Figure_4.22",
    "Figure_4.23",
    "Figure_4.24",
    "Figure_4.25",
    "Figure_4.26",
]

# Loop through each variable and produce a plot of all six iceberg trajectories
i = 0
for var in variable_list:
    print("Processing: %s" % var)
    print("Figure name: %s" % figure_list[i])
    plot_sensitivity_maps(var, figure_list[i])
    i += 1


def plot_sensitivity_maps(variable, figure_name):
    """


    Parameters
    ----------
    variable : str
        Name of variable examined by the sensitivity analysis used to locate
        merged model output files.

    Returns
    -------
    None.

    """
    # Load data
    df = pd.read_csv(
        path_data + "/output/sensitivity_analysis/merged/%s.csv" % variable,
        index_col=False,
    )

    if variable in [
        "length",
        "keel",
        "wind_speed",
        "current_speed",
        "wind_wave_height",
        "swell_wave_height",
    ]:
        label1 = ["Observed", "50%", "75%", "100% (base)", "125%", "150%"]
    elif variable == "added_mass":
        label1 = ["Observed", "0", "0.25", "0.5 (base)", "0.75", "1.0"]
    elif variable == "air_drag":
        label1 = ["Observed", "0.5", "1.0", "1.5", "1.9 (base)", "2.0", "2.5"]
    elif variable == "water_drag":
        label1 = ["Observed", "0.5", "1.0", "1.3 (base)", "1.5", "2.0", "2.5"]
    elif variable in [
        "wind_direction",
        "current_direction",
        "wind_wave_direction",
        "swell_wave_direction",
    ]:
        label1 = ["Observed", "+60°", "+30°", "0° (base)", "-30°", "-60°"]
    # Subset
    df1 = df[df["id"] == "2017_300234060270020"]
    df2 = df[df["id"] == "2017_300234060272000"]
    df3 = df[df["id"] == "2018_300234066241900"]
    df4 = df[df["id"] == "2018_300434063416060"]
    df5 = df[df["id"] == "2019_1124-2670293"]
    df6 = df[df["id"] == "2019_2013-2670502"]

    # Create figure and add axes object (nrows x ncols)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    # To do: Add below lineplots to loop to reduce redundancy
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        color="black",
        data=df1,
        label="Observed",
        ci=None,
        sort=False,
        ax=axs[0, 0],
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        hue="branch",
        style="branch",
        data=df1,
        ci=None,
        sort=False,
        ax=axs[0, 0],
    )
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        color="black",
        data=df2,
        label="Observed",
        ci=None,
        sort=False,
        ax=axs[0, 1],
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        hue="branch",
        style="branch",
        data=df2,
        ci=None,
        sort=False,
        ax=axs[0, 1],
    )
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        color="black",
        data=df3,
        label="Observed",
        ci=None,
        sort=False,
        ax=axs[0, 2],
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        hue="branch",
        style="branch",
        data=df3,
        ci=None,
        sort=False,
        ax=axs[0, 2],
    )
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        color="black",
        data=df4,
        label="Observed",
        ci=None,
        sort=False,
        ax=axs[1, 0],
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        hue="branch",
        style="branch",
        data=df4,
        ci=None,
        sort=False,
        ax=axs[1, 0],
    )
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        color="black",
        data=df5,
        label="Observed",
        ci=None,
        sort=False,
        ax=axs[1, 1],
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        hue="branch",
        style="branch",
        data=df5,
        ci=None,
        sort=False,
        ax=axs[1, 1],
    )
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        color="black",
        data=df6,
        label="Observed",
        ci=None,
        sort=False,
        ax=axs[1, 2],
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        hue="branch",
        style="branch",
        data=df6,
        ci=None,
        sort=False,
        ax=axs[1, 2],
    )
    # Ravel axes to a flattened array
    for ax in axs.ravel():
        ax.grid(ls="dotted")
        ax.set(box_aspect=1, aspect=1, xlabel=None, ylabel=None)
        ax.get_legend().remove()
    fig.supxlabel("East (km)")
    fig.supylabel("North (km)")
    fig.legend(
        title=str(variable).capitalize().replace("_", " "),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        labels=label1,
    )
    axs = axs.flat
    for n, ax in enumerate(axs):
        ax.text(
            -0.125,
            -0.125,
            "(" + string.ascii_lowercase[n] + ")",
            transform=ax.transAxes,
            size=24,
            weight="bold",
        )
    # Save figure
    fig.savefig(
        path_figures + "%s.eps" % figure_name,
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
    )
    plt.close()


# -----------------------------------------------------------------------------
# Figure 5.1
# Map of speed oscillations near Resolution Island
# Last confirmed working 2022-01-18
# -----------------------------------------------------------------------------

# Load data
df = pd.read_csv(
    "/Volumes/data/cis_iceberg_beacon_database/data/2018/300434063415110/standardized_data/2018_300434063415110.csv",
    index_col=False,
)

# Normalize colourbar
norm = mpl.colors.Normalize(vmin=0, vmax=6)
cmap = cm.get_cmap("turbo", 20)

# Set map projection
proj = ccrs.epsg(3347)

# Set extents north of 60°N
extents = [-65.5, -63, 60.5, 62.6]

# Plot
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(
    projection=ccrs.LambertConformal((0.5 * (x1 + x2)), (0.5 * (y1 + y2)))
)  # Centre of extents
ax.add_feature(coast)
ax.set_extent(extents)
# ax.set(box_aspect=1)
ax.set_adjustable("datalim")
sns.scatterplot(
    x="longitude",
    y="latitude",
    hue="speed",
    data=df,
    s=50,
    edgecolor="black",
    palette=cmap,
    legend=False,
    transform=ccrs.PlateCarree(),
)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    x_inline=False,
    y_inline=False,
    alpha=0.25,
    linestyle="dotted",
)
gl.xlocator = mticker.FixedLocator(np.arange(-90, -50, 1))
gl.ylocator = mticker.FixedLocator(np.arange(40, 90, 0.5))
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False

ax.text(
    -66.25,
    62.5,
    "Frobisher Bay",
    fontsize=12,
    weight="bold",
    transform=ccrs.PlateCarree(),
)

ax.text(
    -65.4,
    61.51,
    "Resolution \n    Island",
    fontsize=12,
    weight="bold",
    transform=ccrs.PlateCarree(),
)
"""
ax.annotate(
    text="Resolution \n    Island",
    xy=(-120, 5),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
)
ax.annotate(
    text="Frobisher Bay",
    xy=(-180, 180),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
)
"""
# Add colour bar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("Speed ($\mathrm{km h^{-1}}$)")

# Save figure
fig.savefig(
    path_figures + "Figure_5.1.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -------------------------------------------------------------------------
# Figure 5.2
# Scatterplot of RMSE as a function of iceberg keel depth
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load data
df1 = load_data(2009)
df2 = load_data(2017)

# Change current models to uppercase
df1["current"] = df1["current"].str.upper()
df2["current"] = df2["current"].str.upper()

# Calculate RMSE for each hindcast
stats_track_2009 = (
    df1.groupby(["ensemble", "current"])["dist_error"]
    .apply(rmse)
    .reset_index(name="rmse")
)
stats_track_2017 = (
    df2.groupby(["ensemble", "current"])["dist_error"]
    .apply(rmse)
    .reset_index(name="rmse")
)

# Load iceberg dimensions
df3 = pd.read_csv(path_data + "/input/subset.csv", index_col=False)

# Change column names
df3 = df3.rename(columns={"beacon_id": "id"})

# Select required columns
df4 = df3[["id", "length", "keel"]]

# Merge dimensions
df_2009 = pd.merge(df1, df4, on="id")
df_2017 = pd.merge(df2, df4, on="id")

# Add id columns
df4["id"] = df4["id"].str.split("_").str[1]

# Extract id from ensemble column
stats_track_2009["id"] = stats_track_2009["ensemble"].str.split("_").str[1]
stats_track_2017["id"] = stats_track_2017["ensemble"].str.split("_").str[1]

# Merge stats based on id
temp_2009 = pd.merge(df4, stats_track_2009, on="id")
temp_2017 = pd.merge(df4, stats_track_2017, on="id")

# Plot looping regplots 2009 & 2017
_, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)
for d, m, ls in zip(temp_2009["current"].unique(), ["o", "X", "s"], ["-", "--", "-."]):
    sns.regplot(
        x="keel",
        y="rmse",
        data=temp_2009.loc[temp_2009.current == d],
        marker=m,
        scatter_kws={"linewidths": 1, "edgecolor": "k"},
        line_kws={"ls": ls},
        ax=axs[0],
        label=d,
    )
axs[0].set(xlabel="Keel depth (m)", ylabel="RMSE (km)", xlim=(0, 300))
axs[0].set(box_aspect=1)
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(50))

# axs[0].legend(loc=1)

for d, m, ls in zip(temp_2017["current"].unique(), ["o", "X", "s"], ["-", "--", "-."]):
    sns.regplot(
        x="keel",
        y="rmse",
        data=temp_2017.loc[temp_2017.current == d],
        marker=m,
        scatter_kws={"linewidths": 1, "edgecolor": "k"},
        line_kws={"ls": ls},
        ax=axs[1],
        label=d,
    )
axs[1].set(xlabel="Keel depth (m)", ylabel="RMSE (km)", xlim=(0, 300))
axs[1].set(box_aspect=1)
axs[1].yaxis.set_tick_params(labelleft=True)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(50))
sns.despine()
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(
        -0.15,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=25,
        weight="bold",
    )

plt.savefig(
    path_figures + "Figure_5.2.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -------------------------------------------------------------------------
# Figure 5.3
# Pairwise comparison of distance error
# Last confirmed working 2022-01-13
# -------------------------------------------------------------------------

# CECOM/GLORYS

# Load data
df = load_data(2009)

# CECOM
df1 = df[df["current"] == "cecom"]
df1["segment"] = df1["branch"].str.split("_").str[0]
df1["ensemble"] = df1["id"] + "_" + df1["segment"]

# GLORYS
df2 = df[df["current"] == "glorys"]
df2["segment"] = df2["branch"].str.split("_").str[0]
df2["ensemble"] = df2["id"] + "_" + df2["segment"]

# Pairwise comparison
cecom = df1[["datetime", "ensemble", "dur_obs", "dist_error"]]
glorys = df2[["datetime", "ensemble", "dist_error"]]
merge1 = pd.merge(cecom, glorys, on=["datetime", "ensemble"])
merge1["delta"] = merge1["dist_error_x"] - merge1["dist_error_y"]

# Load data
df = load_data(2017)

# CECOM
df1 = df[df["current"] == "cecom"]
df1["segment"] = df1["branch"].str.split("_").str[0]
df1["ensemble"] = df1["id"] + "_" + df1["segment"]

# GLORYS
df2 = df[df["current"] == "glorys"]
df2["segment"] = df2["branch"].str.split("_").str[0]
df2["ensemble"] = df2["id"] + "_" + df2["segment"]

# RIOPS
df3 = df[df["current"] == "riops"]
df3["segment"] = df3["branch"].str.split("_").str[0]
df3["ensemble"] = df3["id"] + "_" + df3["segment"]

# Pairwise comparison
cecom = df1[["datetime", "ensemble", "dur_obs", "dist_error"]]
glorys = df2[["datetime", "ensemble", "dist_error"]]
riops = df3[["datetime", "ensemble", "dist_error"]]

# Merge
merge2 = cecom.merge(glorys, on=["datetime", "ensemble"], how="left").merge(
    riops, on=["datetime", "ensemble"], how="left"
)

# Calculate difference
merge2["delta_CG"] = merge2["dist_error_x"] - merge2["dist_error_y"]
merge2["delta_CR"] = merge2["dist_error_x"] - merge2["dist_error"]
merge2["delta_GR"] = merge2["dist_error_y"] - merge2["dist_error"]

# Plot
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5), constrained_layout=True)
axs[0].set(xlabel="Hindcast Duration (hours)", ylabel="Distance error (km)")
axs[0].set(box_aspect=1)
axs[0].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
sns.set_palette("turbo", merge1["ensemble"].nunique())
sns.lineplot(
    x="dur_obs",
    y="delta",
    data=merge1,
    hue="ensemble",
    alpha=0.75,
    lw=1,
    ci=None,
    ax=axs[0],
    legend = False,
)
axs[0].axhline(0, ls="-", color="black", alpha=0.75, zorder=10)
axs[1].set(
    xlabel="Hindcast Duration (hours)",
)
axs[1].set(box_aspect=1)
axs[1].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
axs[1].annotate(
    text="Outlier",
    xy=(75, -75),
    xycoords="data",
    fontsize=18,
    weight="bold",
    textcoords="offset points",
    xytext=(-25, -30),
    arrowprops=dict(arrowstyle="->", color="black"),
    va="center",
    ha="center",
)
sns.set_palette("turbo", merge2["ensemble"].nunique())
sns.lineplot(
    x="dur_obs",
    y="delta_CR",
    data=merge2,
    hue="ensemble",
    alpha=0.75,
    lw=1,
    ci=None,
    ax=axs[1],
    legend = False,
)
axs[1].axhline(0, ls="-", color="black", alpha=0.75, zorder=10)
axs[1].yaxis.set_tick_params(labelleft=True)
axs[2].set(xlabel="Hindcast Duration (hours)")
axs[2].set(box_aspect=1)
axs[2].xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
sns.lineplot(
    x="dur_obs",
    y="delta_GR",
    data=merge2,
    hue="ensemble",
    alpha=0.75,
    lw=1,
    ci=None,
    ax=axs[2],
    legend = False,
)
axs[2].axhline(0, ls="-", color="black", alpha=0.75, zorder=10)
axs[2].yaxis.set_tick_params(labelleft=True)
sns.despine()
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.125,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
    if n == 0:
        ax.text(
            0.03,
            0.95,
            "GLORYS better",
            #"$\mathregular{GLORYS_{error}}$",
            transform=ax.transAxes,
            size=16,
            weight="bold",
            ha="left",
        )
        ax.text(
            0.03,
            0.03,
            "CECOM better",
            #"$\mathregular{CECOM_{error}}$",
            transform=ax.transAxes,
            size=16,
            weight="bold",
            ha="left",
        )
    if n == 1:
        ax.text(
            0.03,
            0.95,
            "RIOPS better",
            #"$\mathregular{RIOPS_{error}}$",
            transform=ax.transAxes,
            size=16,
            weight="bold",
            ha="left",
        )
        ax.text(
            0.03,
            0.03,
            "CECOM better",
            #"$\mathregular{CECOM_{error}}$",
            transform=ax.transAxes,
            size=16,
            weight="bold",
            ha="left",
        )
    if n == 2:
        ax.text(
            0.03,
            0.95,
            "GLORYS better",
            #"$\mathregular{GLORYS_{error}}$",
            transform=ax.transAxes,
            size=16,
            weight="bold",
            ha="left",
        )
        ax.text(
            0.03,
            0.03,
            "RIOPS better",
            #"$\mathregular{RIOPS_{error}}$",
            transform=ax.transAxes,
            size=16,
            weight="bold",
            ha="left",
        )
# Save figure
fig.savefig(
    path_figures + "Figure_5.3_test.png", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -----------------------------------------------------------------------------
# Figure 5.4
# Outlier example and ocean currents
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load outlier example data
df = pd.read_csv(
    path_data + "output/validation/error/2017-2019/2017_300234060177480_0.csv",
    index_col=False,
)

# Drop nan rows
df = df.dropna()

# Convert to datetime
df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Add duration column
df["dur_obs"] = df["dur_obs"].astype(int)

# Add ocean current model column
df["current"] = df["branch"].str.split("_").str[1]

# Set map centres
x = df["lon_obs"].median()
y = df["lat_obs"].median()

# Plot
plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.Orthographic(x, y))
ax.add_feature(coast)
ax.set_adjustable("datalim")
lon_formatter = LongitudeFormatter(number_format=".1f")
lat_formatter = LatitudeFormatter(number_format=".1f")
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.3,
    linestyle="dotted",
    xformatter=lon_formatter,
    yformatter=lat_formatter,
)
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False
sns.lineplot(
    x="lon_obs",
    y="lat_obs",
    color="black",
    data=df,
    label="Observed",
    markers=True,
    markevery=3,
    dashes=False,
    ci=None,
    sort=False,
    transform=ccrs.PlateCarree(),
)
sns.lineplot(
    x="lon_model",
    y="lat_model",
    hue="ensemble",
    style="ensemble",
    markers=True,
    markevery=3,
    dashes=False,
    data=df,
    ci=None,
    sort=False,
    transform=ccrs.PlateCarree(),
)
ax.legend(labels=label1)
ax.annotate(
    text="Start",
    xy=(-75, -75),
    xycoords="data",
    fontsize=18,
    weight="bold",
    textcoords="offset points",
)
ax.text(
    -0.125,
    -0.075,
    "(" + string.ascii_lowercase[0] + ")",
    transform=ax.transAxes,
    size=24,
    weight="bold",
)
# Save figure
plt.savefig(
    path_figures + "Figure_5.4a.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# Load data
filename = path_data + "output/validation/model/2017_300234060177480_0.csv"
df = pd.read_csv(filename, index_col=False)

# Convert to datetime
df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Add ocean current model column
df["current"] = df["branch"].str.split("_").str[1]
df["current"] = df["current"].str.upper()

# Subset
df1 = df[df["current"] == "CECOM"].reset_index(drop=True)
df2 = df[df["current"] == "GLORYS"].reset_index(drop=True)
df3 = df[df["current"] == "RIOPS"].reset_index(drop=True)

# Plot quivers
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=True)
axs[0].quiver(
    df1["datetime"],
    0,
    df1["u_current"],
    df1["v_current"],
    color="red",
    zorder=10,
    label="CECOM",
    scale=1,
    scale_units="xy",
    lw=1,
    edgecolor="k",
)
axs[0].set(xlabel="Datetime", ylabel="Speed (m/s)", ylim=(-0.3, 0.3))
axs[1].quiver(
    df2["datetime"],
    0,
    df2["u_current"],
    df2["v_current"],
    color="lime",
    zorder=10,
    label="GLORYS",
    scale=1,
    scale_units="xy",
    lw=1,
    edgecolor="k",
)
axs[1].set(xlabel="Datetime", ylabel="Speed (m/s)")
axs[2].quiver(
    df3["datetime"],
    0,
    df3["u_current"],
    df3["v_current"],
    color="blue",
    zorder=10,
    label="RIOPS",
    scale=1,
    scale_units="xy",
    lw=1,
    edgecolor="k",
)
axs[2].set(xlabel=None, ylabel="Speed (m/s)")
axs[2].xaxis.set_major_locator(mdates.HourLocator(interval=12))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
axs[0].legend(loc=1)
axs[1].legend(loc=1)
axs[2].legend(loc=1)
axs[2].text(
    -0.125,
    -0.3,
    "(" + string.ascii_lowercase[1] + ")",
    transform=axs[2].transAxes,
    size=24,
    weight="bold",
)

fig.autofmt_xdate(rotation=45)
sns.despine()
# Save figure
fig.savefig(
    path_figures + "Figure_5.4b2.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)


# -----------------------------------------------------------------------------
# Figure 5.5
# Boxplots of cumulative distance and lineplots of ocean current speeds
# Last confirmed working 2022-01-13
# -----------------------------------------------------------------------------

# Load base parameter RMSE values
df1 = pd.read_csv(
    path_data + "output/sensitivity_analysis/merged/keel.csv", index_col=False
)

# Add parameter column
df1["parameter"] = df1["branch"].str.split("_").str[0]

replacement_mapping_dict = {
    "0.5": "50%",
    "0.75": "75%",
    "1.0": "100%",
    "1.25": "125%",
    "1.5": "150%",
}

df1["parameter"] = df1["parameter"].replace(replacement_mapping_dict)
df1 = df1.sort_values("datetime").groupby("ensemble").tail(1)

# Load data
df2 = pd.read_csv(
    path_data
    + "/output/sensitivity_analysis/model/swell_wave_height/2018_300234066241900_0.csv",
    index_col=False,
)

# Change datetime
df2["datetime"] = pd.to_datetime(
    df2["datetime"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

# Add parameter column
df2["parameter"] = df2["branch"].str.split("_").str[0]

df2["current_speed_1"] = np.sqrt(df2["u_current1"] ** 2 + df2["v_current1"] ** 2)
df2["current_speed_5"] = np.sqrt(df2["u_current5"] ** 2 + df2["v_current5"] ** 2)
df2["current_speed_10"] = np.sqrt(df2["u_current10"] ** 2 + df2["v_current10"] ** 2)
df2["current_speed_15"] = np.sqrt(df2["u_current15"] ** 2 + df2["v_current15"] ** 2)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
sns.boxplot(x="parameter", y="dist_csum_model", data=df1, ax=axs[0])
plt.setp(axs[0].artists, edgecolor="k", facecolor="w")
plt.setp(axs[0].lines, color="k")
axs[0].set(xlabel="Length Modifier", ylabel="Cumulative Distance (km)")

sns.lineplot(
    x="datetime", y="current_speed_1", data=df2, label="0-10 m", color="red", ax=axs[1]
)
sns.lineplot(
    x="datetime",
    y="current_speed_5",
    data=df2,
    ls="dashed",
    label="40-50 m",
    color="lime",
    ax=axs[1],
)
sns.lineplot(
    x="datetime",
    y="current_speed_10",
    data=df2,
    ls="dotted",
    label="90-100 m",
    color="blue",
    ax=axs[1],
)
sns.lineplot(
    x="datetime",
    y="current_speed_15",
    data=df2,
    ls="dashdot",
    label="140-150 m",
    color="magenta",
    ax=axs[1],
)
axs[1].set(xlabel="Time", ylabel="Speed ($\mathrm{ms^{-1}}$)")
axs[1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.autofmt_xdate(rotation=45)
sns.despine()
for n, ax in enumerate(axs):
    ax.text(
        -0.125,
        -0.4,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
# Save figure
fig.savefig(
    path_figures + "Figure_5.5.eps", dpi=dpi, transparent=False, bbox_inches="tight"
)

# ----------------------------------------------------------------------------
# Appendix B
# Batch plot model output maps using a distance grid instead of lat/lon
# Last confirmed working 2022-01-18 
# ----------------------------------------------------------------------------

# Specify path to model outputs as either:
path_input = path_data + "output/validation/error/2009-2016/"
# or:
path_input = path_data + "output/validation/error/2017-2019/"

# Specify path to output figures
path_figures = path_data + "output/validation/figures/appendix/"

# Find all files in folder
files = glob.glob(path_input + "*.csv")

# Iterate through each error file and create map
for file in files:

    fname = os.path.splitext(os.path.basename(file))[0]

    # Plot error
    plot_distance_maps(fname, path_input, path_figures)

    # Debug
    break


def plot_distance_maps(filename, path_input, path_figures):
    """

    Produce map of modelled and observed iceberg trajectories as distance
    measurements instead of latitude/longitude coordiantes. 
    Distances are specific to relative UTM zone.
    
    Parameters
    ----------
    filename : str
        Filename of model ouput.
    path_input : str
        Path to model outputs.
    path_figures : str
        Path to output figures.

    Returns
    -------
    None.

    """

    # Split filename
    year, beacon, interval = filename.split("_")
    beacon_id = "%s_%s_%s" % (year, beacon, interval)

    # Load model output iceberg trajectory
    df = pd.read_csv(path_input + filename + ".csv", index_col=False)

    # Change column to string
    df["branch"] = df["branch"].astype(str)

    # Add ocean current model column
    df["current"] = df["branch"].str.split("_").str[1]

    # Get final distance error
    distance_error = df.groupby("current")["dist_error"].tail(1)
    #print(distance_error)

    # Calculate RMSE
    stats_track = pd.DataFrame()
    stats_track = df.groupby(['ensemble','current'])['dist_error'].apply(rmse).reset_index(name='rmse')

    # Adjust labels according to number of unique environmental input data sources
    if df["branch"].nunique() == 1:
        label1 = ["Observed", "RIOPS"]
    if df["branch"].nunique() == 2:
        label1 = ["Observed", "CECOM", "GLORYS"]
        dashes = [(5, 1), (1, 1)]
        #title = 
    elif df["branch"].nunique() == 3:
        label1 = ["Observed", "CECOM", "GLORYS", "RIOPS"]
        dashes = [(5, 1), (1, 1), (3, 1, 1, 1)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(ls="dotted")
    ax.set_adjustable("datalim")
    ax.set_aspect("equal")
    ax.set(xlabel="East (km)", ylabel="North (km)")
    if df["branch"].nunique() == 2:
        ax.text(x=0.125, y=0.95, s=f"ID: {beacon}_{interval} Start: {df.datetime.min()} Duration: {df.dur_obs.max():.0f} hrs", fontsize=18, fontweight="bold", ha="left", transform=fig.transFigure)
        ax.text(x=0.125, y=0.92, s=f"{'DE (km):':<15} CECOM: {distance_error.iloc[0]:.1f} GLORYS: {distance_error.iloc[1]:.1f}", fontsize=18, ha="left", transform=fig.transFigure)
        ax.text(x=0.125, y=0.89, s=f"{'RMSE (km):':<12} CECOM: {stats_track.rmse.iloc[0]:.1f} GLORYS: {stats_track.rmse.iloc[1]:.1f}", fontsize=18, ha="left", transform=fig.transFigure)

    elif df["branch"].nunique() == 3:
        ax.text(x=0.125, y=0.95, s=f"ID: {beacon}_{interval} Start: {df.datetime.min()} Duration: {df.dur_obs.max():.0f} hrs", fontsize=18, fontweight="bold", ha="left", transform=fig.transFigure)
        ax.text(x=0.125, y=0.92, s=f"{'DE (km):':<15} CECOM: {distance_error.iloc[0]:.1f} GLORYS: {distance_error.iloc[1]:.1f} RIOPS: {distance_error.iloc[2]:.1f}", fontsize=18, ha="left", transform=fig.transFigure)
        ax.text(x=0.125, y=0.89, s=f"{'RMSE (km):':<12} CECOM: {stats_track.rmse.iloc[0]:.1f} GLORYS: {stats_track.rmse.iloc[1]:.1f} RIOPS: {stats_track.rmse.iloc[2]:.1f}", fontsize=18, ha="left", transform=fig.transFigure)

        
    sns.lineplot(
        x="x_obs",
        y="y_obs",
        data=df,
        color="black",
        label="Observed",
        ci=None,
        sort=False,
    )
    sns.lineplot(
        x="x_model",
        y="y_model",
        data=df,
        hue="branch",
        style="branch",
        dashes=dashes,
        ci=None,
        sort=False,
    )
    ax.legend(labels=label1)
    fig.savefig(
        path_figures + "map_%s.eps" % filename,
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
        format='eps',
    )
    plt.close()
    
    
    
    
    
'''
    
    ax.text(
        -0.125,
        -0.175,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        size=24,
        weight="bold",
    )
  
            
        f"The comedian is {comedian['name']}, aged {comedian['age']}."
        
        ax.text(x=0.125, y=0.945, s="ID: %s-%s Start: %s Dur: %0.0f hrs" % (beacon, interval, df["datetime"].min(), df["dur_obs"].max()), fontsize=18, fontweight="bold", ha="left", transform=fig.transFigure)
        #ax.text(x=0.125, y=0.89, s="DE (km): CECOM: %0.1f GLORYS: %0.1f RIOPS: %0.1f\nRMSE (km): CECOM: %0.1f GLORYS: %0.1f RIOPS: %0.1f" % (distance_error.iloc[0], distance_error.iloc[1], distance_error.iloc[2],stats_track["rmse"].iloc[0],stats_track["rmse"].iloc[1],stats_track["rmse"].iloc[2]), fontsize=18, ha="left", transform=fig.transFigure)
        ax.text(x=0.125, y=0.89, s=f"{'DE (km): ' + distance_error.iloc[0]:<10} GLORYS: {distance_error.iloc[1]}", fontsize=18, ha="left", transform=fig.transFigure)
        #plt.subplots_adjust(top=0.8, wspace=0.3)
test = 1


        ax.set_title(
            r"$\bf{%s-%s %s %s hours\nDE (km):      CECOM: %0.1f GLORYS: %0.1f RIOPS: %0.1f\nRMSE (km): CECOM: %0.1f GLORYS: %0.1f RIOPS: %0.1f}$"
            % (beacon, interval, df["datetime"].min(), df["dur_obs"].max(), distance_error.iloc[0], distance_error.iloc[1], distance_error.iloc[2],
               stats_track["rmse"].iloc[0],stats_track["rmse"].iloc[1],stats_track["rmse"].iloc[2]),
            loc="left",
            fontsize=18,
            #fontweight="bold",
        )     
'''