#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:42:18 2022

@author: adam
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'mean_direction_of_total_swell', 'mean_direction_of_wind_waves', 'mean_period_of_total_swell',
            'mean_period_of_wind_waves', 'significant_height_of_total_swell', 'significant_height_of_wind_waves',
        ],
        'year': '2021',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            85, -100, 40, -40,
        ],
        'format': 'netcdf',
    },
    'era5_waves_2021.nc')