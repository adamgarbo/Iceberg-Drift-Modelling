"""
riopsrotator.py

Created on Sun Feb  7 12:23:03 2021

Last modified: Jan 3 2022

@author: Adam Garbo

Description: 
    Rotates and interpolates Regional Ice Ocean Prediction System (RIOPS) 
    modelled ocean current data.
    Steps:
    1) Rotate U and V vectors
    2) Merge of individual T, U and V files
    3) Interpolate depth levels to 0 - 200 m
    4) Interpolate to latitude/longitude grid
    5) Save netCDF file with compression enabled


Lay out assumptions and dependencies of the code - libraries, directory structure file formats, etc.     

Usage:    
python riopsrotator.py -h 

Example of processing by iterating through files: 
python riopsrotator.py /Users/adam/Desktop/temp/python_test output angle_file.nc gridfile_0_05.txt Creg12-CMC-ANAL_1h_ICEBERG_T_*.nc.gz

Example of processing a single file:
python riopsrotator.py /Users/adam/Desktop/temp/python_test output angle_file.nc gridfile_0_05.txt Creg12-CMC-ANAL_1h_ICEBERG_T_20170701.nc.gz

Example of parallel processing
- Navigate to the working directory

find . -name '*_T_*.nc.gz' -type f | parallel -j 14 --nice 15 --progress python riopsrotator.py /Users/adam/Desktop/temp/python_test output angle_file.nc gridfile_0_05.txt 

To access the functions in the file, it can be importaned from another *.py file if it is in the PATH
This is a good reason to keep the functions generic. e.g.,: 
import riopsrotator

temperatures = riopsrotator.readnc(Creg12-CMC-ANAL_1h_ICEBERG_T_20170701.nc.gz)

Notes:
    Python code formatted using Black:
    https://github.com/psf/black
    
"""

import os
import glob
import sys
import time
import gzip
import shutil
import tempfile
import argparse
import numpy as np
import xarray as xr
from cdo import *
import logging
from datetime import datetime

# Logfile path
path = "/Volumes/data/nais_iceberg_drift_model/riops/test/"

# Create logfile with timestamp
file = datetime.now().strftime("log_%Y%m%d_%H%M%S.csv")

# Join path and filename
logfile = os.path.join(path, file)

# Configure logging to terminal and logfile
format = "%(asctime)s,%(message)s"
handlers = [logging.FileHandler(logfile, mode="w"), logging.StreamHandler()]

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    format=format, handlers=handlers, datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
)

# Create logfile header row
logging.debug("duration,task,description")

# Debugging
filename = "/Volumes/data/nais_iceberg_drift_model/riops/test/Creg12-CMC-ANAL_1h_ICEBERG_T_20180101-20180101.nc.gz"
# theta = xr.open_dataset('/Volumes/data/nais_iceberg_drift_model/riops/scripts/angle_file.nc').LAAN.isel(x=slice(397,617), y=slice(282,1070)).squeeze()
# gridfile = '/Volumes/data/nais_iceberg_drift_model/riops/scripts/gridfile_1_12.txt'


def open_netcdf(filename):
    """
    Read RIOPS compressed NetCDF (*.nc.gz) file and load data using xarray.
    Note: Uncompressed NetCDF files can also be read.

    See: https://stackoverflow.com/questions/27322248/how-to-read-a-gzip-netcdf-file-in-python/27325655

    Parameters
    ----------
    filename : str
        Relative or absolute path to RIOPS file.

    Returns
    -------
    data : xarray Dataset

    """

    if filename.endswith(".gz"):
        # Open gzip-compressed file in binary mode
        infile = gzip.open(filename, "rb")
        # Create temporary file object
        temp = tempfile.NamedTemporaryFile(delete=False)
        print("File created: ", temp)
        print("Filename: ", temp.name)
        # Copy gzip file object to temporary file object
        shutil.copyfileobj(infile, temp)
        # Close gzip reader
        infile.close()
        # Close temporary file and delete file object
        temp.close()
        # Open NetCDF dataset from temporary file object
        data = xr.open_dataset(temp.name)
        # Unlink temporary file object
        os.unlink(temp.name)
    elif filename.endswith(".nc"):
        data = xr.open_dataset(filename)
    return data


def check_rotation(ds, theta, x_loc, y_loc, time_loc, depth_loc):
    """
    Check rotated U and V components against manual equation calculations.

    Parameters
    ----------
    ds : xarray Dataset
        DESCRIPTION.
    theta : xarray DataArray
        DESCRIPTION.
    x_loc : int
        DESCRIPTION.
    y_loc : int
        DESCRIPTION.
    time_loc : int
        DESCRIPTION.
    depth_loc : int
        DESCRIPTION.

    Returns
    -------
    None.

    """

    u_rot = ds["uo"].isel(
        x=x_loc, y=y_loc, time=time_loc, depth=depth_loc
    ).values * np.cos(theta.isel(x=(x_loc), y=(y_loc)).values) - ds["vo"].isel(
        x=(x_loc), y=(y_loc), time=time_loc, depth=depth_loc
    ).values * np.sin(
        theta.isel(x=(x_loc), y=(y_loc)).values
    )
    v_rot = ds["vo"].isel(
        x=x_loc, y=y_loc, time=time_loc, depth=depth_loc
    ).values * np.cos(theta.isel(x=(x_loc), y=(y_loc)).values) + ds["uo"].isel(
        x=(x_loc), y=(y_loc), time=time_loc, depth=depth_loc
    ).values * np.sin(
        theta.isel(x=(x_loc), y=(y_loc)).values
    )

    print(
        "Rotated:\t",
        ds["uo_rot"].isel(x=(x_loc), y=(y_loc), time=time_loc, depth=depth_loc).values,
        ds["vo_rot"].isel(x=(x_loc), y=(y_loc), time=time_loc, depth=depth_loc).values,
    )
    print("Check:\t", u_rot, v_rot)


def rotate_interp(filename, out_dir, theta, gridfile):
    """
    Rotate and interpolate T, U and V RIOPS data files

    Parameters
    ----------
    filename : str
        Relative or absolute path to RIOPS nc.gz file (i.e., T [for Temperature])
        Note: Assumes there is a corresponding U and V file.
    out_dir : str
        Path to output directory.
    theta : xarray object
        An xarray object with rotation angles (must be a certain size?).
    grid : str
        Name of the grid file to resample to (follow conventions for this).

    Returns
    -------
    None

    """
    # Start debugging script timer
    start_time = time.time()

    logging.info("%.2f,filename,%s" % ((time.time() - start_time), filename))

    # Filenames
    file_t = filename  # T
    file_u = filename.replace("_T_", "_U_")  # U
    file_v = filename.replace("_T_", "_V_")  # V
    file_out = filename.replace("_T_", "_ROTVEL2_")  # Output filename
    file_out = os.path.splitext(file_out)[0]  # Remove .gz extension
    file_out = os.path.join(out_dir, file_out)  # Add output directory path to filename

    # Load data (t, u and v files)
    ds_t = open_netcdf(file_t)
    ds_u = open_netcdf(file_u)
    ds_v = open_netcdf(file_v)

    # Rename coordinates to avoid duplication during arithmetic operations
    ds_t = ds_t.rename({"deptht": "depth", "time_counter": "time"})
    ds_u = ds_u.rename({"depthu": "depth", "time_counter": "time"})
    ds_v = ds_v.rename({"depthv": "depth", "time_counter": "time"})

    # Extract desired variables into DataArray and slice depth to upper 250 m
    da_t = ds_t.thetao.sel(depth=slice(0, 250))
    da_u = ds_u.uo.sel(depth=slice(0, 250))
    da_v = ds_v.vo.sel(depth=slice(0, 250))

    # Record attributes
    u_attrs = da_u.attrs
    v_attrs = da_v.attrs

    # Remove unnecessary coordinates
    da_t = da_t.drop_vars(["time_instant"])
    da_u = da_u.drop_vars(["nav_lat", "nav_lon", "time_instant"])
    da_v = da_v.drop_vars(["nav_lat", "nav_lon", "time_instant"])

    # Merge DataArrays into a DataSet
    ds = xr.merge([da_t, da_u, da_v])

    # Log script execution time for data loading
    logging.info("%.2f, load_data " % (time.time() - start_time))

    # Average adjacent U and V component values (time, depth, y, x)
    ds["uo"][:, :, :, 1:] = 0.5 * (ds["uo"][:, :, :, :-1] + ds["uo"][:, :, :, 1:])
    ds["vo"][:, :, 1:, :] = 0.5 * (ds["vo"][:, :, :-1, :] + ds["vo"][:, :, 1:, :])

    # Rotate U and V components and store results in temporary file
    ds_u_tmp = ds["uo"] * np.cos(theta) - ds["vo"] * np.sin(theta)
    ds_v_tmp = ds["vo"] * np.cos(theta) + ds["uo"] * np.sin(theta)

    # Insert averaged and rotated U and V variables into T file
    ds["uo"] = ds_u_tmp
    ds["vo"] = ds_v_tmp

    # Log script execution time for rotation
    logging.info("%.2f, rotation" % (time.time() - start_time))

    # Optional: Save to NetCDF file
    # ds.to_netcdf("../riops/test/rotated.nc")
    # logging.info("%.2f, save" % (time.time() - start_time))

    # Create object for CDO Python bindings
    cdo = Cdo()

    # Interpolate depth levels to centres of 10 m layers
    ds_int = cdo.intlevel(
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
        105,
        115,
        125,
        135,
        145,
        155,
        165,
        175,
        185,
        195,
        input=ds,
        returnXDataset=True,
    )

    # Log script execution time for interpolation
    logging.info("%.2f, interpolation" % (time.time() - start_time))

    # Reassign attribute information to coordiantes
    ds_int.uo.attrs = u_attrs
    ds_int.vo.attrs = v_attrs
    ds_int.depth.attrs = {
        "units": "m",
        "valid_min": 0.0,
        "valid_max": 200.0,
        "long_name": "Depth at T-grid points",
        "standard_name": "depth",
        "axis": "Z",
        "positive": "down",
    }

    # Interpolate to latitude-longitude grid and save as compressed NetCDF file
    # ds_remap = cdo.remapbil(gridfile, input = ds_int, returnXDataset = True)
    cdo.remapbil(gridfile, input=ds_int, output=file_out, options="-z zip")

    # Log execution time for remapping
    logging.info("%.2f, remapping" % (time.time() - start_time))

    # Clean temporary files
    cdo.cleanTempDir()

    # Log execution time for clearing temporary files
    logging.info("%.2f, clear_temp" % (time.time() - start_time))


def main():

    # Create object instance of the ArgumentParser class
    parser = argparse.ArgumentParser()

    # Add argument requirements to execute the script
    parser.add_argument("path_input", help="Enter input path to raw RIOPS data.")
    parser.add_argument("path_output", help="Enter output path.")
    parser.add_argument("angle_file", help="Enter angle file name")
    parser.add_argument("grid_file", help="Enter grid file name")
    parser.add_argument(
        "nc_file",
        help="Enter RIOPS T .nc.gz file name or pattern to loop through.",
    )
    args = parser.parse_args()

    path_input = args.path_input
    path_output = args.path_output
    angle_file = args.angle_file
    grid_file = args.grid_file
    nc_file = args.nc_file

    # Debugging: Uncomment for testing locally
    # in_dir ='.../riops/test'
    # out_dir = '.../riops/test'
    # ang_file = '.../riops/scripts/angle_file.nc'
    # grd_file = '.../riops/scripts/gridfile_1_12.txt'
    # nc_file  = 'Creg12-CMC-ANAL_1h_ICEBERG_T_*.nc.gz'

    # Set working directory
    os.chdir(in_dir)

    # Create output directory
    if not os.path.isdir(out_dir):
        try:
            os.mkdir(out_dir)
        except OSError:
            print("Unable to create directory: %s " % out_dir)
            sys.exit()
    print("Created directory: %s " % out_dir)

    if not os.path.isfile(ang_file):
        print("Angle file not available")
        sys.exit()

    # Load angle file based on subdomain corresponding to U, V, and T files
    # E.g., [398:618, 283:788]
    # Note: Slicing is hard coded and should be made into arguments
    theta = (
        xr.open_dataset(ang_file)
        .LAAN.isel(x=slice(397, 617), y=slice(282, 1070))
        .squeeze()
    )

    if not os.path.isfile(grd_file):
        print("Error: Grid file not available")
        sys.exit()

    if os.path.isfile(nc_file):
        print("Found a single RIOPS file")
        rotate_interp(nc_file, out_dir, theta, grd_file)
    else:
        # Assume if it is not matching that NetCDF file is a file pattern
        # Identify all files in the current directory matching "_T_" filename wildcard
        files = glob.glob(nc_file)
        # Iterate through each "_T_" file
        for filename in files:
            rotate_interp(filename, out_dir, theta, grd_file)


if __name__ == "__main__":
    main()
