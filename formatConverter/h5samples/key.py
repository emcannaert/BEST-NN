import h5py
import numpy as np

def get_bes_vars_as_array(h5_file, dataset_name):
    with h5py.File(h5_file, 'r') as f:
        data = np.array(f[dataset_name])
        size = f[dataset_name].shape
    return data, size

# Usage
h5_file = 'bg_2015.h5'
dataset_name = 'BES_vars'
bes_vars_array, size = get_bes_vars_as_array(h5_file, dataset_name)

print("Dataset size:", size)
print(bes_vars_array)

output_file_path = "2015.txt"
with open(output_file_path, 'w') as outfile:
            outfile.write(bes_vars_array)
        

