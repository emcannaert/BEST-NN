import os
import h5py
import numpy as np

# Directory containing the .h5 files
directory = '.'

# Directory containing the .txt files
txt_directory = 'txt_files'

# Output file
output_file = 'bg_combined.h5'

# Get a list of all .h5 files in the directory that start with 'allDecays'
file_paths = [f for f in os.listdir(directory) if f.endswith('train_1.h5')]

# Get a list of all .txt files in the txt directory
txt_file_paths = [os.path.join(txt_directory, f) for f in os.listdir(txt_directory) if f.endswith('.txt')]

# Read the fractions from the txt files
fractions = {}
for txt_file_path in txt_file_paths:
    with open(txt_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            ht = float(parts[1])
            qcd_fraction = float(parts[2])
            tt_fraction = float(parts[3])
            W_fraction = float(parts[4])
            ST_fraction = float(parts[5])
            fractions[ht] = (qcd_fraction, tt_fraction, W_fraction, ST_fraction)

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

            # Apply the fractions to the QCD and TT samples
            ht_values = data[:, 111]  # Assuming the second column is HT
            for i, ht in enumerate(ht_values):
                if ht in fractions:
                    qcd_fraction, tt_fraction, W_fraction, ST_fraction = fractions[ht]
                    if ((qcd_fraction == 0) and (tt_fraction == 0) and (W_fraction == 0) and (ST_fraction == 0)):
                        qcd_fraction = 0.99
                        tt_fraction = 0.5

                        data[i, :] *= qcd_fraction + tt_fraction

            end = start + data.shape[0]
            combined[start:end, :] = data
            start = end

        print('Processed {}'.format(file_path))

# Open the output file in read mode and print the shape and dtype of the dataset
with h5py.File(output_file, 'r') as f:
    print(f['BES_vars'].shape)
    print(f['BES_vars'].dtype)

print('Done')