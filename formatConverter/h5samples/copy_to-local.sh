#!/bin/bash

# Define the source directory
src_dir="root://cmseos.fnal.gov//store/user/tjian/h5_updated/MChi"

# Use a for loop to iterate over each file in the source directory
for file in $(xrdfs root://cmseos.fnal.gov/ ls /store/user/tjian/h5_updated/MChi| grep allDecays_MChi.*); do
    # Use xrdcp to copy each file to the same directory structure in the destination
    xrdcp -f "root://cmseos.fnal.gov/${file}" .
done