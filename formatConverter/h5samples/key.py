import h5py
import numpy as np

def get_bes_vars_as_array(h5_file, dataset_name):
    with h5py.File(h5_file, 'r') as f:
        data = np.array(f[dataset_name])
        size = f[dataset_name].shape
    return data, size

# Usage
h5_file = 'allDecays_Sample_all_mass_combine_BESTinputs_test_flattened_standardized.h5'
dataset_name = 'BES_vars'
bes_vars_array, size = get_bes_vars_as_array(h5_file, dataset_name)

print("Dataset size:", size)
print(bes_vars_array)