#!/bin/bash

# Define the source directory
src_dir="root://cmseos.fnal.gov//store/user/tjian/h5_updated/"

# Use a for loop to iterate over each file in the source directory
for file in $(ls .| grep .*bg); do
    # Use xrdcp to copy each file to the same directory structure in the destination
    xrdcp -f "${file}" "${src_dir}" 
done