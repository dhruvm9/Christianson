#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:55:43 2023

@author: dhruv
"""
import numpy as np
import os
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import NeuroScopeRecordingInterface

#%% 

data_directory = '/media/dhruv/Ultra Touch/Dhruv-Jamie/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

s = datasets[0]

#%% 

# For Neuroscope we need to pass the location of the `.dat` file
file_path = os.path.join(data_directory, s, s + '.dat')
# Change the file_path to the location in your system
interface = NeuroScopeRecordingInterface(file_path=file_path, verbose=False)

# Extract what metadata we can from the source files
metadata = interface.get_metadata()
# session_start_time is required for conversion. If it cannot be inferred
# automatically from the source files you must supply one.
session_start_time = datetime.today() #(2023, 9, 5, 13, 53, 2, tzinfo=tz.gettz("US/Eastern"))
metadata["NWBFile"].update(session_start_time=session_start_time)

 # Choose a path for saving the nwb file and run the conversion
nwbfile_path = os.path.join(data_directory, s, 'pynapplenwb', s + '.nwb')  # This should be something like: "./saved_file.nwb"
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)