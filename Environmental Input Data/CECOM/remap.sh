#!/bin/bash
# Automated script to remap CECOM data

# Requires Climate Data Operators tools (CDO)
# conda install -c conda-forge cdo

# Path to compressed CECOM files 
CECOM_DIR=/Volumes/data/nais_iceberg_drift_model/cecom

# Location of the grid file used for remapping
GRID=/Volumes/data/nais_iceberg_drift_model/cecom/scripts/gridfile.txt    

# Change to process multiple years at once
#years=(2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019)
years=(2019)

for year in ${years[@]}
do
    echo "Processing: $year"

    DATA_DIR=$CECOM_DIR/$year
    EXTRACT_DIR=$DATA_DIR/extract
    REMAP_DIR=$DATA_DIR/remap
    TAR_DIR=$DATA_DIR/tar
    NC_DIR=$DATA_DIR/nc

    echo "DATA_DIR = $DATA_DIR"
    echo "EXTRACT_DIR = $EXTRACT_DIR"
    echo "REMAP_DIR = $REMAP_DIR"
    echo "TAR_DIR = $TAR_DIR"
    echo "NC_DIR = $NC_DIR"

    # Check if extract directory exist
    if [ ! -d $EXTRACT_DIR ];
    then
        mkdir $EXTRACT_DIR
        echo "Created $EXTRACT_DIR"
    else
        echo "Error: Directory $EXTRACT_DIR already exists"
    fi

    # Check if remap directory exists
    if [ ! -d $REMAP_DIR ];
    then
        mkdir $REMAP_DIR
        echo "Created $REMAP_DIR"
    else
        echo "Error: Directory $REMAP_DIR already exists"
    fi

    # Check if remap directory exists
    if [ ! -d $TAR_DIR ];
    then
        mkdir $TAR_DIR
        echo "Created $TAR_DIR"
    else
        echo "Error: Directory $TAR_DIR already exists"
    fi

    # Check if remap directory exists
    if [ ! -d $NC_DIR ];
    then
        mkdir $NC_DIR
        echo "Created $NC_DIR"
    else
        echo "Error: Directory $NC_DIR already exists"
    fi

    # Uncompress .tar files
    for file in $DATA_DIR/*.tar
    do
        tar xvf $file -C $DATA_DIR
    done

    # Uncompress .nc.gz files
    gunzip $DATA_DIR -v -r

    # Move unextracted files from their respective folders
    find $DATA_DIR -name "*.nc" -exec mv '{}' $NC_DIR \;

    # Move .tar files into a folder
    find $DATA_DIR -name "*.tar" -exec mv '{}' $TAR_DIR \;

    # Extract required variables
    for file in $NC_DIR/*.nc
    do
        echo "Extracting variables from: ${file##*/}"
        cdo -f nc4 select,timestep=3/10,name=OceanCurrentU,OceanCurrentV,OceanTemp $file $EXTRACT_DIR/${file##*/}
    done

    # Delete temporary .nc files
    rm -rf $NC_DIR

    # Remap files to centered 10 m depth levels
    for file in $EXTRACT_DIR/*.nc
    do
        echo "Remapping: ${file##*/}"
        cdo -f nc4 -remapbil,$GRID -intlevel,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195 $file $REMAP_DIR/${file##*/}
    done

    # Delete temporary extracted .nc files
    rm -rf $EXTRACT_DIR

    # Merge with compression enabled and in netCDF4 classic format
    #cdo -z zip mergetime $REMAP_DIR/*.nc $DATA_DIR/cecom_merge_"$year".nc
    
    # Concatenate the remapped data (vastly more efficient than merging)
    cdo -z zip cat $REMAP_DIR/*.nc $DATA_DIR/cecom_"$year".nc

    # Delete temporary remapped .nc files once concatenation is complete
    rm -rf $REMAP_DIR

    # Delete any empty folders
    find $DATA_DIR -type 'd' -exec rmdir '{}' \;

done
