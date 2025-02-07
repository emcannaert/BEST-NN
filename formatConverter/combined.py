import os
import h5py
import numpy as np

# Directory containing the .h5 files
directory = '.'

# Output file
output_file = 'allDecays_combined.h5'

# Get a list of all .h5 files in the directory that start with 'allDecays'
file_paths = [f for f in os.listdir(directory) if f.endswith('train_1.h5') and f.startswith('allDecays')]

# Determine the total size of the new dataset
total_size = 0
for file_path in file_paths:
    with h5py.File(file_path, 'r') as f:
        total_size += f['BES_vars'].shape[0]

# Create a new h5 file and an empty dataset
with h5py.File(output_file, 'w') as f:
    combined = f.create_dataset('BES_vars', (total_size, 182), dtype='float32')

    # Loop over all the files and append their data
    start = 0
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f_in:
            data = f_in['BES_vars'][...]
            end = start + data.shape[0]
            combined[start:end, :] = data
            start = end

        print('Processed {}'.format(file_path))

# Open the output file in read mode and print the shape and dtype of the dataset
with h5py.File(output_file, 'r') as f:
    print(f['BES_vars'].shape)
    print(f['BES_vars'].dtype)

print('Done')